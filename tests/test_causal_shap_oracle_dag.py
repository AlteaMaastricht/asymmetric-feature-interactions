import unittest

import networkx as nx
import numpy as np

import _path  # noqa: F401
from asym_interactions import scm
from asym_interactions.methods.causal_shap_oracle_dag import (
    dag_ordered_shapley_products,
    topological_orderings,
)


def additive_model(X):
    """No interactions at all."""
    return 1.0 * X[:, 0] + 2.0 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + 1.0 * X[:, 4]


def scm_target(X):
    return scm.structural_target(*X.T)


class TestTopologicalOrderings(unittest.TestCase):
    def test_default_dag_has_seven_orderings_and_all_are_valid(self):
        orders = topological_orderings(scm.DAG_EDGES, 5, max_orderings=100)
        self.assertEqual(len(orders), 7)
        self.assertEqual(len(orders), len(list(nx.all_topological_sorts(nx.DiGraph(scm.DAG_EDGES)))))
        for o in orders:
            for i, j in scm.DAG_EDGES:
                self.assertLess(o.index(i), o.index(j))

    def test_cycle_rejected(self):
        with self.assertRaises(ValueError):
            topological_orderings([(0, 1), (1, 0)], 2)


class TestDagOrderedShapley(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, y, GT, edges = scm.generate_asymmetric_dataset(dict(scm.DEFAULT_CONFIG, n_samples=400), seed=0)
        cls.X = df.values
        cls.edges = edges

    def _run(self, f, edges):
        return dag_ordered_shapley_products(f, self.X, edges, n_samples=30, max_orderings=100, background_size=50, seed=0)

    def test_direction_is_copied_from_the_given_dag(self):
        """For every ancestor pair in the DAG, the reverse entry is exactly zero."""
        M = self._run(scm_target, self.edges)
        G = nx.DiGraph(self.edges)
        for j in G.nodes:
            for i in nx.ancestors(G, j):
                self.assertGreater(M[i, j], 0)
                self.assertEqual(M[j, i], 0.0)

    def test_reversed_dag_reverses_every_edge(self):
        rev = [(j, i) for i, j in self.edges]
        M = self._run(scm_target, rev)
        for i, j in self.edges:
            self.assertEqual(M[i, j], 0.0)
            self.assertGreater(M[j, i], 0)

    def test_nonzero_for_additive_model_without_interactions(self):
        """The entry is a product of marginal contributions, not an interaction measure."""
        M = self._run(additive_model, self.edges)
        for i, j in self.edges:
            self.assertGreater(M[i, j], 0.1)

    def test_shape_and_diagonal(self):
        M = self._run(scm_target, self.edges)
        self.assertEqual(M.shape, (5, 5))
        np.testing.assert_array_equal(np.diag(M), np.zeros(5))


if __name__ == "__main__":
    unittest.main()
