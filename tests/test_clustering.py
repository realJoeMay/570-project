"""Clustering preparation, quality metrics, and within-cluster neighbor rules."""

import unittest

import numpy as np
import pandas as pd

from utils.clustering import evaluate_kmeans, prepare_clustering_data, select_cluster_count, within_cluster_neighbors


class ClusteringTests(unittest.TestCase):
    def test_selection_adapts_when_previous_six_cluster_choice_is_not_shortlisted(self):
        metrics = pd.DataFrame({
            "k": [2, 6, 9], "silhouette": [0.295, 0.276, 0.298],
            "davies_bouldin": [1.355, 1.138, 1.073],
            "peer_pool_ok": [True, True, False], "stable": [True, True, False],
            "near_best_silhouette": [True, False, True],
        })
        selected, reason = select_cluster_count(metrics)
        self.assertEqual(selected, 2)
        self.assertIn("current feasible", reason)
        selected, reason = select_cluster_count(metrics, preferred_k=6)
        self.assertEqual(selected, 6)
        self.assertIn("outside", reason)
        with self.assertRaises(ValueError):
            select_cluster_count(metrics, preferred_k=9)

    def test_empty_silhouette_shortlist_uses_feasible_candidate(self):
        metrics = pd.DataFrame({
            "k": [2, 3], "silhouette": [0.2, 0.4], "davies_bouldin": [1.2, 1.0],
            "peer_pool_ok": [True, False], "stable": [True, True],
            "near_best_silhouette": [False, True],
        })
        selected, reason = select_cluster_count(metrics)
        self.assertEqual(selected, 2)
        self.assertIn("No feasible count met", reason)
        metrics["stable"] = False
        with self.assertRaisesRegex(ValueError, "No tested cluster count"):
            select_cluster_count(metrics)

    def test_preprocessing_fits_complete_records_and_preserves_missingness(self):
        frame = pd.DataFrame({
            "parish_index": [0, 1, 2, 3, 4],
            "county_population": [100, 1000, 10000, 1e12, np.nan],
            "county_median_age": [30, 40, 50, 60, np.nan],
            "n_social_links": [0, 2, 4, np.nan, np.nan],
            "attached_school": [False, True, False, pd.NA, pd.NA],
        })
        original = frame.copy()
        transformed, matrix, complete, quality, preprocessor = prepare_clustering_data(frame)
        self.assertEqual(complete.tolist(), [True, True, True, False, False])
        np.testing.assert_allclose(matrix[:3, :3].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_array_equal(matrix[:, 3], [0, 1, 0, 0, 0])
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(quality.n_observed_features.tolist(), [4, 4, 4, 2, 0])
        self.assertEqual(quality.iloc[3].imputed_features, "n_social_links, attached_school")
        self.assertAlmostEqual(transformed.iloc[0].log_county_population, np.log1p(100))
        pd.testing.assert_frame_equal(frame, original)
        scaler = preprocessor.named_transformers_["continuous"].named_steps["scaler"]
        self.assertAlmostEqual(scaler.mean_[1], 40)  # Partial record must not influence fitting.

    def test_metrics_favor_three_well_separated_groups(self):
        rng = np.random.default_rng(0)
        matrix = np.vstack([rng.normal(center, 0.1, size=(20, 2)) for center in [-5, 0, 5]])
        scores, models = evaluate_kmeans(matrix, [2, 3, 100], seeds=[1, 2], n_init=5)
        self.assertEqual(scores.k.tolist(), [2, 3])
        scores = scores.set_index("k")
        self.assertGreater(scores.loc[3, "silhouette"], scores.loc[2, "silhouette"])
        self.assertLess(scores.loc[3, "davies_bouldin"], scores.loc[2, "davies_bouldin"])
        self.assertEqual(scores.loc[3, "min_cluster_size"], 20)
        self.assertAlmostEqual(scores.loc[3, "mean_seed_ari"], 1)
        self.assertEqual(set(models), {2, 3})

    def test_neighbors_exclude_self_partial_peers_and_other_clusters(self):
        parishes = pd.DataFrame({
            "parish_index": [10, 12, 11, 13, 14, 15],
            "name": ["Target", "Tie B", "Tie A", "Partial target", "Other cluster", "No data"],
            "diocese": ["Mobile", "Other", "Other", "Mobile", "Other", "Mobile"],
            "county_fips": ["01001", "01003", "01001", "01003", "01003", pd.NA],
            "cluster": pd.Series([0, 0, 0, 0, 1, pd.NA], dtype="Int64"),
            "is_reference": [True, True, True, False, True, False],
            "assignment_quality": ["complete", "complete", "complete", "provisional_imputed", "complete", "unassigned_no_features"],
            "imputed_features": ["", "", "", "attached_school", "", "all"],
        })
        matrix = np.array([[0, 0], [1, 0], [-1, 0], [0.1, 0], [0.01, 0], [0, 0]])
        matches, summary = within_cluster_neighbors(parishes, matrix, target_diocese="Mobile", top_n=10)
        target = matches.loc[matches.target_parish_index == 10]
        self.assertEqual(target.peer_parish_index.tolist(), [11, 12])
        self.assertEqual(target["rank"].tolist(), [1, 2])
        np.testing.assert_allclose(target.distance, [1, 1])
        self.assertEqual(target.same_county.tolist(), [True, False])
        self.assertNotIn(13, matches.peer_parish_index)
        self.assertNotIn(14, matches.peer_parish_index)
        self.assertEqual(len(summary), 3)
        self.assertEqual(summary.set_index("target_parish_index").loc[15, "match_status"], "unassigned")
        self.assertEqual(summary.set_index("target_parish_index").loc[10, "match_status"], "fewer_than_requested")
        self.assertTrue(matches.target_parish_index.ne(matches.peer_parish_index).all())
        empty_matches, empty_summary = within_cluster_neighbors(parishes, matrix, target_diocese="Absent")
        self.assertEqual(len(empty_matches), 0)
        self.assertIn("target_parish_index", empty_matches)
        self.assertEqual(len(empty_summary), 0)


if __name__ == "__main__":
    unittest.main()
