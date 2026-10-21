"""Evidence-preserving ministry comparisons and per-target report coverage."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.parish_reports import (
    generate_parish_reports, ministry_coverage, neighbor_ministry_gaps,
    summarize_ministries, validate_parish_rows,
)


class ParishReportTests(unittest.TestCase):
    def setUp(self):
        self.parishes = pd.DataFrame([
            {"parish_index": i, "name": name, "diocese": diocese, "site_url": f"https://parish{i}.org/", "cluster": cluster}
            for i, name, diocese, cluster in [(1, "Target", "Target diocese", 0), (2, "Neighbor", "Other", 0), (3, "No data", "Target diocese", "")]
        ])
        rows = [(1, "Music Ministry", "ok", "music"), (1, "music ministry!", "ok", "choir"),
                (2, "MUSIC MINISTRY", "ok", "music"), (2, "Food Pantry", "ok", "pantry"),
                (2, "Food Pantry", "ok", "food"), (2, "Unconfirmed label", "error", "failed")]
        self.results = pd.DataFrame([
            {**self.parishes.set_index("parish_index").loc[i].to_dict(), "parish_index": i,
             "ministry": label, "category": "Service", "page_url": f"https://parish{i}.org/{page}", "prediction_status": status}
            for i, label, status, page in rows
        ])
        self.matches = pd.DataFrame([{
            "target_parish_index": 1, "target_name": "Target", "peer_parish_index": 2,
            "peer_name": "Neighbor", "peer_site_url": "https://parish2.org/", "rank": 1, "distance": 0.2, "cluster": 0,
        }])
        self.targets = pd.DataFrame([
            {"target_parish_index": 1, "target_name": "Target", "cluster": 0, "assignment_quality": "complete", "imputed_features": "", "n_matches": 1, "match_status": "fewer_than_requested"},
            {"target_parish_index": 3, "target_name": "No data", "cluster": "", "assignment_quality": "unassigned_no_features", "imputed_features": "", "n_matches": 0, "match_status": "unassigned"},
        ])

    def test_identity_check_rejects_reassigned_row_ids(self):
        stale = self.results.copy()
        stale.loc[0, "parish_index"] = 2
        valid, rejected = validate_parish_rows(stale, self.parishes)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(valid), len(stale) - 1)

    def test_ministries_deduplicate_labels_keep_evidence_and_exclude_errors(self):
        ministries = summarize_ministries(self.results)
        self.assertEqual(len(ministries), 3)
        target = ministries.loc[ministries.parish_index.eq(1)].iloc[0]
        self.assertEqual(target.n_source_pages, 2)
        self.assertEqual(len(json.loads(target.source_urls)), 2)
        self.assertNotIn("Unconfirmed label", ministries.ministry.tolist())

    def test_gap_comparison_does_not_count_duplicates_or_common_labels(self):
        ministries = summarize_ministries(self.results)
        gaps = neighbor_ministry_gaps(self.matches, ministries)
        self.assertEqual(gaps.ministry.tolist(), ["Food Pantry"])
        self.assertEqual(gaps.iloc[0].n_source_pages, 2)

    def test_coverage_uses_page_identities_and_preserves_unprocessed_parishes(self):
        ministries = summarize_ministries(self.results)
        pages = self.results.copy()
        pages.loc[0, "page_url"] = "https://parish1.org/not-processed"
        coverage = ministry_coverage(self.parishes, pages, self.results, ministries).set_index("parish_index")
        self.assertEqual(coverage.loc[1, "coverage_status"], "partial")
        self.assertEqual(coverage.loc[1, "n_unresolved_listed_pages"], 1)
        self.assertEqual(coverage.loc[3, "coverage_status"], "no_listed_pages")

    def test_report_written_for_every_target_including_no_data_target(self):
        ministries = summarize_ministries(self.results)
        coverage = ministry_coverage(self.parishes, self.results, self.results, ministries)
        gaps = neighbor_ministry_gaps(self.matches, ministries)
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_parish_reports(
                self.parishes, self.targets, self.matches, ministries, coverage, gaps,
                report_dir=directory, target_diocese="Target diocese",
            )
            self.assertEqual(len(manifest), 2)
            self.assertTrue((Path(directory) / "index.md").exists())
            target = Path(manifest.set_index("target_parish_index").loc[1, "report_path"]).read_text(encoding="utf-8")
            self.assertIn("## Parish information", target)
            self.assertIn("## Current parish ministries", target)
            self.assertIn("## Neighbor descriptions", target)
            self.assertIn("Food Pantry", target)
            self.assertIn("https://parish2.org/food", target)
            empty = Path(manifest.set_index("target_parish_index").loc[3, "report_path"]).read_text(encoding="utf-8")
            self.assertIn("No ministry labels are recorded", empty)
            self.assertIn("No neighbors were returned", empty)
            self.assertEqual(manifest.set_index("target_parish_index").loc[1, "n_additional_ministry_labels"], 1)


if __name__ == "__main__":
    unittest.main()
