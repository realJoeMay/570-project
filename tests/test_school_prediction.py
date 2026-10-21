"""School prompts, parish limits, caching, and missing-value behavior."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from utils.llm import LLMClient
from utils.external_links import extract_homepage_links
from utils.school_prediction import SchoolPrediction, extract_school_features


class SchoolPredictionTests(unittest.TestCase):
    def test_school_prompt_uses_internal_pages_and_external_links(self):
        rows = extract_homepage_links(
            '<a href="/school">Our school</a><a href="https://school.parish0.org/">School site</a>'
            '<a href="https://facebook.com/parish">Facebook</a>',
            "https://parish0.org/", "https://parish0.org/",
        )
        self.links = pd.DataFrame([{**row, "parish_index": 0} for row in rows])
        with patch.object(LLMClient, "_request", return_value='{"attached_school":"yes"}') as request:
            result = self.run_prediction(parish_limit=1)
        links = json.loads(request.call_args.args[0][1]["content"])["homepage_links"]
        self.assertEqual(len(links), 3)
        self.assertEqual(sum(link["link_type"] == "internal" for link in links), 2)
        self.assertIn("/school", [link["path"] for link in links])
        self.assertTrue(result.iloc[0]["attached_school"])
        self.assertEqual(result.iloc[0]["school_extractor_version"], "2.0.0")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.client = LLMClient(provider="ollama", model="test-model", cache_dir=Path(temporary.name))
        self.parishes = pd.DataFrame([
            {"parish_index": i, "name": f"Parish {i}", "diocese": "Test",
             "site_url": f"https://parish{i}.org/", "address": "Town, AL"}
            for i in range(12)
        ])
        self.links = pd.DataFrame([
            {"parish_index": i, "link_url": url, "link_text": label, "link_type": "external"}
            for i in range(12) for url, label in [
                ("https://school.edu/admissions?year=2026", "Our parish school"),
                ("https://facebook.com/parish", "Facebook"),
            ]
        ])
        self.statuses = pd.DataFrame({"parish_index": range(12), "homepage_status": "ok"})

    def run_prediction(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return extract_school_features(
                self.parishes, self.links, self.statuses, llm=self.client, **kwargs,
            )

    def test_limit_ten_and_shared_cache(self):
        with patch.object(LLMClient, "_request", return_value='{"attached_school":"yes"}') as request:
            first = self.run_prediction(parish_limit=10)
            self.assertEqual(request.call_count, 10)
            self.assertTrue(first.iloc[:10]["attached_school"].all())
            self.assertTrue(first.iloc[10:]["attached_school"].isna().all())
            second = self.run_prediction(parish_limit=10)
            self.assertEqual(request.call_count, 10)
            self.assertTrue(second.iloc[:10]["school_prediction_status"].eq("cached").all())
            self.run_prediction(parish_limit=None)
            self.assertEqual(request.call_count, 12)

    def test_prompt_includes_all_links_and_changed_links_invalidate_cache(self):
        with patch.object(LLMClient, "_request", return_value='{"attached_school":"no"}') as request:
            result = self.run_prediction(parish_limit=1)
            messages = request.call_args.args[0]
            payload = json.loads(messages[1]["content"])
            self.assertEqual(len(payload["homepage_links"]), 2)
            school = next(link for link in payload["homepage_links"] if link["url"].startswith("https://school.edu"))
            self.assertEqual(school["path"], "/admissions")
            self.assertEqual(school["link_text"], "Our parish school")
            self.assertFalse(result.iloc[0]["attached_school"])
            self.links.loc[0, "link_url"] = "https://school.edu/new"
            self.run_prediction(parish_limit=1)
            self.assertEqual(request.call_count, 2)

    def test_unavailable_homepage_skips_call_but_empty_links_are_predicted(self):
        self.statuses.loc[0, "homepage_status"] = "cache_missing"
        self.links = self.links.loc[self.links.parish_index != 1]
        with patch.object(LLMClient, "_request", return_value='{"attached_school":"no"}') as request:
            result = self.run_prediction(parish_limit=2)
            request.assert_called_once()
            payload = json.loads(request.call_args.args[0][1]["content"])
            self.assertEqual(payload["homepage_links"], [])
        self.assertTrue(pd.isna(result.iloc[0]["attached_school"]))
        self.assertEqual(result.iloc[0]["school_prediction_status"], "homepage_unavailable")
        self.assertFalse(result.iloc[1]["attached_school"])

    def test_invalid_answer_stays_missing_and_is_retryable(self):
        with patch.object(LLMClient, "_request", return_value='{"attached_school":"maybe"}') as request:
            first = self.run_prediction(parish_limit=1)
            self.run_prediction(parish_limit=1)
            self.assertEqual(request.call_count, 2)
        self.assertTrue(pd.isna(first.iloc[0]["attached_school"]))
        self.assertEqual(first.iloc[0]["school_prediction_status"], "error")
        with self.assertRaises(ValidationError):
            SchoolPrediction(attached_school=True)

    def test_zero_and_invalid_limits(self):
        with patch.object(LLMClient, "_request") as request:
            result = self.run_prediction(parish_limit=0)
            self.assertTrue(result["attached_school"].isna().all())
            request.assert_not_called()
        for limit in [-1, 1.5, True]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.run_prediction(parish_limit=limit)


if __name__ == "__main__":
    unittest.main()
