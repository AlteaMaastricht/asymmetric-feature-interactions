import unittest

import numpy as np

import _path
from asym_interactions import scm
from asym_interactions.methods.pc_dag_mask import apply_dag_mask, directed_mask_from_causallearn_graph


class TestMaskConversion(unittest.TestCase):
    def test_causallearn_encoding(self):
        # causal-learn: i -> j  <=>  G[i, j] == -1 and G[j, i] == 1 ; undirected = -1/-1
        G = np.zeros((5, 5))
        G[0, 2], G[2, 0] = -1, 1   # 0 -> 2
        G[1, 2], G[2, 1] = -1, 1   # 1 -> 2
        G[2, 3], G[3, 2] = -1, 1   # 2 -> 3
        G[1, 4], G[4, 1] = -1, -1  # 1 - 4 undirected (PC cannot orient it in this SCM)
        mask = directed_mask_from_causallearn_graph(G)
        expected = scm.dag_adjacency()
        expected[1, 4] = 0
        np.testing.assert_array_equal(mask, expected)

    def test_masking_a_symmetric_matrix_gives_direction_only_from_mask(self):
        rng = np.random.default_rng(0)
        S = rng.random((5, 5))
        S = S + S.T
        np.fill_diagonal(S, 0)
        M = apply_dag_mask(S, scm.dag_adjacency())
        for i, j in scm.DAG_EDGES:
            self.assertGreater(M[i, j], 0)
            self.assertEqual(M[j, i], 0)
        self.assertEqual(int((M != 0).sum()), len(scm.DAG_EDGES))


@unittest.skipUnless(_path.HAS_CAUSALLEARN, "needs causal-learn")
class TestPCRun(unittest.TestCase):
    def test_pc_recovers_v_structure(self):
        from asym_interactions.methods.pc_dag_mask import pc_directed_graph

        df, *_ = scm.generate_asymmetric_dataset(seed=42)
        mask = pc_directed_graph(df.values[:2250], alpha=0.1, ci_test="fisherz")
        self.assertEqual(mask[0, 2], 1)
        self.assertEqual(mask[1, 2], 1)


if __name__ == "__main__":
    unittest.main()
