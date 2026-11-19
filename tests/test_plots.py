import os
import tempfile
import unittest

import numpy as np

import _path
from asym_interactions import plots, scm
from asym_interactions.metrics import METRIC_KEYS, all_metrics


def fake_inputs():
    GT = scm.ground_truth_matrix()
    rng = np.random.default_rng(0)
    S = rng.random((5, 5)); S = S + S.T; np.fill_diagonal(S, 0)
    mats = {"GT": GT, "IG2": S, "IG2xPC": S * scm.dag_adjacency(), "DAGShapOracle": GT + 0.1 * rng.random((5, 5))}
    means = {m: all_metrics(mats[m], GT) for m in plots.METHODS}
    stds = {m: {k: float("nan") for k in METRIC_KEYS} for m in plots.METHODS}
    mask = ~np.eye(5, dtype=bool)
    roc = {m: [(GT[mask] > 0).astype(int).tolist(), mats[m][mask].tolist()] for m in plots.METHODS}
    return mats, means, stds, roc


class TestMatplotlibOnlyPlots(unittest.TestCase):
    """Synthetic inputs only -- checks that the figure code runs; these are not results."""

    def test_write_figures(self):
        mats, means, stds, roc = fake_inputs()
        with tempfile.TemporaryDirectory() as d:
            plots.plot_asymmetry_comparison(means, stds, os.path.join(d, "a.png"), n_seeds=1)
            plots.plot_edge_recovery_bar(means, stds, os.path.join(d, "b.png"), n_seeds=1)
            plots.plot_roc_pr_curves(roc, os.path.join(d, "c.png"), os.path.join(d, "d.png"))
            plots.plot_directed_graphs(mats, os.path.join(d, "e.png"))
            plots.plot_radar_chart(means, os.path.join(d, "f.png"))
            plots.plot_runtime_complexity([1.0, 2.0, 3.0], [np.nan] * 3, os.path.join(d, "g.png"), n_seeds=1, budgets="x")
            for f in "abcdefg":
                self.assertGreater(os.path.getsize(os.path.join(d, f + ".png")), 1000)


@unittest.skipUnless(_path.HAS_SEABORN, "needs seaborn")
class TestSeabornPlots(unittest.TestCase):
    def test_heatmaps(self):
        mats, *_ = fake_inputs()
        with tempfile.TemporaryDirectory() as d:
            plots.plot_heatmaps(mats, list(mats), os.path.join(d, "h.png"))
            plots.plot_diverging_heatmaps(mats, os.path.join(d, "i.png"))
            plots.plot_rank_consistency(mats, os.path.join(d, "j.png"))


if __name__ == "__main__":
    unittest.main()
