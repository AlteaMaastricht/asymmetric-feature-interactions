import unittest

import numpy as np

import _path
from asym_interactions.experiments import run_compare


class TestCLI(unittest.TestCase):
    """The parser is importable and testable without torch."""

    def test_defaults_match_original_script(self):
        args = run_compare.build_parser().parse_args([])
        config, budgets = run_compare.config_from_args(args)
        self.assertEqual(args.seeds, [42])
        self.assertEqual(args.outdir, "outputs_compare")
        self.assertEqual(config, {"n_samples": 3000, "noise_level": 0.05, "interaction_type": "nonlinear",
                                  "include_confounding": False, "causal_nonlinear": False})
        self.assertEqual(budgets, run_compare.DEFAULT_BUDGETS)
        self.assertEqual((budgets["ig_samples"], budgets["ig_steps"], budgets["ig_baselines"]), (20, 30, 2))
        self.assertEqual((budgets["pc_test"], budgets["pc_alpha"]), ("fisherz", 0.1))
        self.assertEqual((budgets["cs_samples"], budgets["cs_orderings"], budgets["cs_background"]), (200, 100, 150))

    def test_multiple_seeds_and_options(self):
        args = run_compare.build_parser().parse_args(
            ["--seeds", "1", "2", "3", "--confounding", "--interaction-type", "threshold", "--pc-test", "kci"])
        config, budgets = run_compare.config_from_args(args)
        self.assertEqual(args.seeds, [1, 2, 3])
        self.assertTrue(config["include_confounding"])
        self.assertEqual(config["interaction_type"], "threshold")
        self.assertEqual(budgets["pc_test"], "kci")

    def test_std_is_nan_for_single_seed(self):
        self.assertTrue(np.isnan(run_compare._std([1.0])))
        self.assertAlmostEqual(run_compare._std([1.0, 3.0]), np.sqrt(2.0))


@unittest.skipUnless(_path.HAS_TORCH and _path.HAS_CAUSALLEARN, "needs torch + causal-learn (decision D9)")
class TestSmallPipelineRun(unittest.TestCase):
    def test_tiny_run(self):
        from asym_interactions import scm

        cfg = dict(scm.DEFAULT_CONFIG, n_samples=300)
        budgets = dict(run_compare.DEFAULT_BUDGETS, epochs=5, ig_samples=3, ig_baselines=1, ig_steps=10,
                       cs_samples=5, cs_orderings=7, cs_background=20)
        matrices, runtimes, _, metrics, extra = run_compare.run_single_seed(cfg, 0, budgets)
        for m in run_compare.METHODS:
            self.assertEqual(matrices[m].shape, (5, 5))
        self.assertAlmostEqual(runtimes["IG2xPC"], runtimes["IG2"] + runtimes["PC_only"])
        np.testing.assert_allclose(matrices["IG2"], matrices["IG2"].T, atol=1e-3 * max(1e-9, matrices["IG2"].max()) + 1e-6)


if __name__ == "__main__":
    unittest.main()
