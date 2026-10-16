"""Cached page extraction behavior, independent of the notebook UI."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from utils.ministry_extraction import extract_page_ministries
from utils.ministry_prediction import MinistryPrediction


class MinistryExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.llm = SimpleNamespace(provider="ollama", model="test-model", cache_dir=self.root / "cache")
        self.pages = pd.DataFrame([
            {
                "parish_index": 0, "name": "Test", "diocese": "Test",
                "site_url": "https://parish.example", "page_url": f"https://parish.example/{suffix}",
                "request_llm": eligible, "rule_ministry": "", "matched_rule": "" if eligible else "homepage",
            }
            for suffix, eligible in [("", False), ("servers", True), ("lectors", True)]
        ])

    def run_extraction(self, pages=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return extract_page_ministries(
                self.pages if pages is None else pages, llm=self.llm, output_dir=self.output, **kwargs,
            )

    def test_limited_batches_reuse_cache_and_combine_output(self):
        with patch("utils.ministry_extraction.predict_ministry", return_value=MinistryPrediction(ministry=None)) as predict:
            first = self.run_extraction(page_limit=1)
            self.assertEqual(len(first), 2)
            self.assertEqual(predict.call_count, 1)
            second = self.run_extraction(page_limit=1)
            self.assertEqual(len(second), 3)
            self.assertEqual(predict.call_count, 2)
            self.run_extraction(page_limit=1)
            self.assertEqual(predict.call_count, 2)
        exported = pd.read_csv(self.output / "parish_ministries.csv", keep_default_na=False)
        self.assertEqual(len(exported), 3)
        self.assertEqual(set(exported["prediction_status"]), {"rule", "ok"})

    def test_rules_override_cache_and_changed_inputs_rerun(self):
        with patch("utils.ministry_extraction.predict_ministry", return_value=MinistryPrediction(ministry=None)) as predict:
            self.run_extraction(page_limit=None)
            self.assertEqual(predict.call_count, 2)
            changed = self.pages.copy()
            changed.loc[1, "request_llm"] = False
            changed.loc[1, "matched_rule"] = "new exclusion"
            result = self.run_extraction(changed, page_limit=None)
            self.assertEqual(predict.call_count, 2)
            self.assertEqual(result.loc[result["page_url"].str.endswith("servers"), "prediction_status"].iloc[0], "rule")
            changed.loc[2, "page_title"] = "Updated title"
            changed = changed.fillna("")
            self.run_extraction(changed, page_limit=None)
            self.assertEqual(predict.call_count, 3)

    def test_failure_retry_and_previous_cache_migration(self):
        with patch("utils.ministry_extraction.predict_ministry", side_effect=ConnectionError("offline")) as predict:
            result = self.run_extraction(page_limit=1)
            self.assertIn("error", result["prediction_status"].tolist())
            self.run_extraction(page_limit=1)
            self.assertEqual(predict.call_count, 2)
        # First populate the original cache location, then move to the configured folder.
        self.llm.cache_dir = self.output / "cache"
        with patch("utils.ministry_extraction.predict_ministry", return_value=MinistryPrediction(ministry=None)) as predict:
            self.run_extraction(page_limit=None)
            self.assertEqual(predict.call_count, 2)
            self.llm.cache_dir = self.root / "new-cache"
            self.run_extraction(page_limit=None)
            self.assertEqual(predict.call_count, 2)
            self.assertEqual(len(list(self.llm.cache_dir.rglob("*.json"))), 2)

    def test_empty_input_and_invalid_limit(self):
        result = self.run_extraction(self.pages.head(0), page_limit=0)
        self.assertTrue(result.empty)
        self.assertIn("ministry", result.columns)
        for limit in [-1, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.run_extraction(page_limit=limit)


if __name__ == "__main__":
    unittest.main()
