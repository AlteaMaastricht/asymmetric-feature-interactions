import unittest

import numpy as np

import _path
from asym_interactions import scm
from asym_interactions.methods.ig_second_difference import second_difference_reference


class TestReferenceSecondDifference(unittest.TestCase):
    """NumPy reference for what the IG2 matrix converges to (runs without torch)."""

    def test_symmetric_for_scm_target(self):
        rng = np.random.default_rng(0)
        f = lambda v: float(scm.structural_target(*v))  # noqa: E731
        for _ in range(50):
            A = second_difference_reference(f, rng.normal(size=5), rng.normal(size=5))
            np.testing.assert_allclose(A, A.T, atol=1e-12)
            np.testing.assert_array_equal(np.diag(A), np.zeros(5))

    def test_clamp_drops_positive_interactions(self):
        f = lambda v: float(v[0] * v[1])  # noqa: E731
        A_pos = second_difference_reference(f, np.array([1.0, 1.0]), np.zeros(2))   # D = +1
        A_neg = second_difference_reference(f, np.array([1.0, -1.0]), np.zeros(2))  # D = -1
        self.assertEqual(A_pos[0, 1], 0.0)
        self.assertAlmostEqual(A_neg[0, 1], 1.0)

    def test_zero_for_additive_function(self):
        f = lambda v: float(np.sum(np.sin(v)))  # noqa: E731
        A = second_difference_reference(f, np.ones(4), np.zeros(4))
        np.testing.assert_allclose(A, 0.0, atol=1e-12)


@unittest.skipUnless(_path.HAS_TORCH, "needs torch (decision D9)")
class TestTorchIG2(unittest.TestCase):
    def test_matches_reference_and_is_symmetric(self):
        import torch

        from asym_interactions.methods.ig_second_difference import ig_second_difference
        from asym_interactions.model import SmallNet

        torch.manual_seed(0)
        net = SmallNet().eval()
        f = lambda v: float(net(torch.tensor(v, dtype=torch.float32).unsqueeze(0)).item())  # noqa: E731
        rng = np.random.default_rng(0)
        for _ in range(5):
            x, b = rng.normal(size=5), rng.normal(size=5)
            A = ig_second_difference(net, torch.tensor(x, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), n_steps=200)
            ref = second_difference_reference(f, x, b)
            np.testing.assert_allclose(A, ref, atol=1e-3)
            np.testing.assert_allclose(A, A.T, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
