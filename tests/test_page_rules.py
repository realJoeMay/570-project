"""Ordered URL-rule behavior and parish identity preservation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.page_rules import apply_page_rules, load_rules, matches


class PageRulesTests(unittest.TestCase):
    def test_match_types(self):
        for kind, pattern, path in [
            ("homepage", "", ""), ("equals", "/servers/", "servers"),
            ("contains", "server", "ministries/servers"),
            ("starts_with", "people/", "people"),
            ("starts_with", "people", "people/bob"),
            ("ends_with", "servers", "ministries/servers"),
            ("regex", r"^ministry/\d+$", "ministry/12"),
        ]:
            with self.subTest(kind=kind, path=path):
                self.assertTrue(matches({"match_type": kind, "pattern": pattern}, path))
        self.assertFalse(matches({"match_type": "homepage", "pattern": ""}, "about"))

    def test_first_match_decoding_and_default(self):
        rules = [
            {"line": 2, "match_type": "starts_with", "pattern": "people", "ministry": "", "request_llm": False},
            {"line": 3, "match_type": "contains", "pattern": "bob", "ministry": "Candidate", "request_llm": True},
        ]
        pages = pd.DataFrame({"page_url": ["https://a.org/People/%42ob/?x=1#top", "https://a.org/servers"]})
        result = apply_page_rules(pages, rules)
        self.assertEqual(result.loc[0, "path"], "people/bob")
        self.assertFalse(result.loc[0, "request_llm"])
        self.assertEqual(result.loc[0, "matched_rule"], "line 2: starts_with people")
        self.assertTrue(result.loc[1, "request_llm"])
        self.assertEqual(result.loc[1, "matched_rule"], "")

    def test_preserve_shared_urls_and_empty_input(self):
        pages = pd.DataFrame([
            {"parish_index": 0, "site_url": "https://a.org", "page_url": "https://shared.org/servers"},
            {"parish_index": 1, "site_url": "https://b.org", "page_url": "https://shared.org/servers"},
            {"parish_index": 0, "site_url": "https://a.org", "page_url": "https://shared.org/servers"},
        ])
        self.assertEqual(len(apply_page_rules(pages, [])), 2)
        result = apply_page_rules(pages.head(0), [])
        self.assertTrue(result.empty)
        self.assertIn("request_llm", result.columns)

    def test_load_rule_labels_and_booleans(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.csv"
            path.write_text("match_type,pattern,ministry,request_llm\nEQUALS,Servers,Altar Servers,yes\n", encoding="utf-8-sig")
            rules = load_rules(path)
            result = apply_page_rules(pd.DataFrame({"page_url": ["https://a.org/SERVERS"]}), rules)
            self.assertEqual(result.loc[0, "rule_ministry"], "Altar Servers")
            self.assertTrue(result.loc[0, "request_llm"])
            path.write_text("match_type,pattern,ministry,request_llm\ninvalid,x,,false\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rules(path)


if __name__ == "__main__":
    unittest.main()
