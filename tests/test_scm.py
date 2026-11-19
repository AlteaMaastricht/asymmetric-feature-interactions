import unittest

import numpy as np

import _path  # noqa: F401
from asym_interactions import scm


class TestSCM(unittest.TestCase):
    def test_shapes_and_columns(self):
        df, y, GT, edges = scm.generate_asymmetric_dataset(seed=0)
        self.assertEqual(df.shape, (3000, 5))
        self.assertEqual(list(df.columns), scm.FEATURE_NAMES)
        self.assertEqual(y.shape, (3000,))
        self.assertEqual(GT.shape, (5, 5))
        self.assertEqual(edges, [(0, 2), (1, 2), (2, 3), (1, 4)])

    def test_seed_reproducible_and_seed_sensitive(self):
        a = scm.generate_asymmetric_dataset(seed=42)
        b = scm.generate_asymmetric_dataset(seed=42)
        c = scm.generate_asymmetric_dataset(seed=43)
        np.testing.assert_array_equal(a[0].values, b[0].values)
        np.testing.assert_array_equal(a[1], b[1])
        self.assertFalse(np.allclose(a[0].values, c[0].values))

    def test_ground_truth_support_equals_dag_edges(self):
        """The evaluation target is the DAG: GT is non-zero exactly on the DAG edges."""
        GT = scm.ground_truth_matrix()
        np.testing.assert_array_equal((GT != 0).astype(float), scm.dag_adjacency())
        self.assertTrue(np.all(GT[GT.T != 0] == 0), "GT is strictly one-directional")

    def test_target_second_differences_are_symmetric(self):
        """Mixed second differences of y are equal in both orders, so GT's asymmetry is not a property of y."""
        rng = np.random.default_rng(1)
        for itype in scm.INTERACTION_TYPES:
            f = lambda v: scm.structural_target(*v, interaction_type=itype)  # noqa: E731
            for _ in range(20):
                x, b = rng.normal(size=5), rng.normal(size=5)
                d = x - b
                for i in range(5):
                    for j in range(i + 1, 5):
                        def shift(*ks):
                            p = b.copy()
                            for k in ks:
                                p[k] += d[k]
                            return f(p)
                        Dij = shift(i, j) - shift(i) - shift(j) + f(b)
                        Dji = shift(j, i) - shift(j) - shift(i) + f(b)
                        self.assertAlmostEqual(Dij, Dji, places=10)

    def test_default_dag_structure_matches_regressions(self):
        """With no confounding, X0 and X1 are independent; X2 depends on both; X4 on X1."""
        df, *_ = scm.generate_asymmetric_dataset(seed=7)
        C = np.corrcoef(df.values.T)
        self.assertLess(abs(C[0, 1]), 0.06)
        self.assertGreater(C[0, 2], 0.4)
        self.assertGreater(C[1, 2], 0.3)
        self.assertGreater(C[2, 3], 0.8)
        self.assertGreater(C[1, 4], 0.6)
        self.assertLess(abs(C[0, 4]), 0.06)

    def test_confounding_option_correlates_roots(self):
        cfg = dict(scm.DEFAULT_CONFIG, include_confounding=True)
        df, *_ = scm.generate_asymmetric_dataset(cfg, seed=7)
        self.assertGreater(np.corrcoef(df["X0"], df["X1"])[0, 1], 0.5)

    def test_noise_level_scales_residual(self):
        cfg = dict(scm.DEFAULT_CONFIG, noise_level=0.05)
        df, y, *_ = scm.generate_asymmetric_dataset(cfg, seed=3)
        clean = scm.structural_target(*df.values.T)
        ratio = np.var(y - clean) / np.var(clean)
        self.assertAlmostEqual(ratio, 0.05, delta=0.01)

    def test_unknown_interaction_type(self):
        with self.assertRaises(ValueError):
            scm.generate_asymmetric_dataset(dict(scm.DEFAULT_CONFIG, interaction_type="bogus"), seed=0)


if __name__ == "__main__":
    unittest.main()
