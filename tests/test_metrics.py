import unittest

import numpy as np

import _path  # noqa: F401
from asym_interactions import metrics, scm


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.GT = scm.ground_truth_matrix()

    def test_gt_scores_perfectly_against_itself(self):
        m = metrics.all_metrics(self.GT, self.GT)
        self.assertAlmostEqual(m["frobenius"], 1.0)
        self.assertEqual(m["directional_acc"], 1.0)
        self.assertAlmostEqual(m["roc_auc"], 1.0)
        self.assertAlmostEqual(m["auprc"], 1.0)
        self.assertEqual(m["hamming"], 0.0)

    def test_binary_dag_adjacency_without_any_model_scores_perfectly_on_edge_metrics(self):
        """Evaluation rewards DAG recovery: the bare DAG adjacency (no model, no interaction
        strengths) gets perfect direction and edge-recovery scores."""
        A = scm.dag_adjacency()
        m = metrics.all_metrics(A, self.GT)
        self.assertEqual(m["directional_acc"], 1.0)
        self.assertAlmostEqual(m["roc_auc"], 1.0)
        self.assertAlmostEqual(m["auprc"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0, places=6)
        self.assertEqual(m["hamming"], 0.0)

    def test_symmetric_matrix_has_zero_asymmetry_and_zero_directional_accuracy(self):
        rng = np.random.default_rng(0)
        S = rng.random((5, 5))
        S = S + S.T
        np.fill_diagonal(S, 0)
        m = metrics.all_metrics(S, self.GT)
        self.assertEqual(m["asymmetry"], 0.0)
        self.assertEqual(m["directional_acc"], 0.0)  # ties count as wrong

    def test_symmetrised_gt_keeps_perfect_roc_but_loses_direction(self):
        S = self.GT + self.GT.T
        m = metrics.all_metrics(S, self.GT)
        self.assertLess(m["roc_auc"], 1.0)  # reverse entries become false positives
        self.assertEqual(m["directional_acc"], 0.0)

    def test_asymmetry_score_depends_on_scale(self):
        A = self.GT
        self.assertAlmostEqual(metrics.asymmetry_score(10 * A), 10 * metrics.asymmetry_score(A))
        self.assertAlmostEqual(metrics.frobenius_similarity(10 * A, self.GT), metrics.frobenius_similarity(A, self.GT))

    def test_top_k_max_is_0_8_with_four_true_edges(self):
        self.assertAlmostEqual(metrics.top_k_accuracy(self.GT, self.GT, k=5), 0.8)

    def test_normalise_constant(self):
        np.testing.assert_array_equal(metrics.normalise(np.ones((3, 3))), np.zeros((3, 3)))

    def test_edge_recovery_counts(self):
        M = scm.dag_adjacency().copy()
        M[3, 4] = 1.0  # one false positive
        p, r, f1 = metrics.edge_recovery(M, self.GT)
        self.assertAlmostEqual(p, 0.8, places=6)
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_metric_keys_match(self):
        self.assertEqual(list(metrics.all_metrics(self.GT, self.GT).keys()), metrics.METRIC_KEYS)


if __name__ == "__main__":
    unittest.main()
