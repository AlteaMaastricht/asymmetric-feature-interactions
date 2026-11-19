"""Integrated-Gradients second-difference interaction matrix ("IG2").

Origin: ``detect_claude.py`` l.989-1098, where the functions were called
``asymmetric_integrated_gradients`` / ``compute_aig_matrix`` and the result was
plotted as "Integrated Hessians". It is neither asymmetric nor Integrated
Hessians (Janizek et al., 2021). The computation is unchanged; only the names
are honest now.

What it computes
----------------
For an explicand ``x`` and a baseline ``b`` (both standardised), with
``d = x - b`` and ``e_k`` the k-th unit vector, each single-coordinate
line integral of the gradient equals a finite difference (fundamental theorem
of calculus, up to trapezoidal quadrature error):

    ig_first[j]     = int_0^1 df/dx_j (b + a d_j e_j) da * d_j  ~ f(b + d_j e_j) - f(b)
    ig_second[i, j] = (same, starting from b + d_i e_i)          ~ f(b + d_i e_i + d_j e_j) - f(b + d_i e_i)

    A[i, j] = max(0, ig_first[i] - ig_second[j, i])
            = max(0, -D_ij)   with   D_ij = f(b+d_i e_i+d_j e_j) - f(b+d_i e_i) - f(b+d_j e_j) + f(b)

``D_ij`` is the mixed second difference of ``f``, which is **symmetric in i and
j by construction** (D_ij == D_ji). Any asymmetry in ``A`` is quadrature noise
(about 1e-5 of the matrix range in the original run; see
``results/original_detect_claude_2026-05-31/fig7_diverging_heatmaps.png``).

Note also the ``max(0, .)`` clamp: only *negative* (sub-additive) second
differences are kept; positive (super-additive) interaction is set to zero.
``compute_ig_second_difference_matrix`` averages ``A`` over explicands and
baselines, i.e. it estimates E[max(0, -D_ij)].

:func:`second_difference_reference` gives the exact (quadrature-free) value of
``A`` for any NumPy function, and is used by the tests.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:  # torch is optional at import time so the NumPy reference stays usable
    import torch
except ImportError:  # pragma: no cover - depends on environment
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("This function needs PyTorch (pip install torch).")


def _line_integral_of_gradient(model, baseline, target, feature_index: int, n_steps: int = 50):
    """Single line integral of df/dx_feature from baseline to target.

    Kept from the original file (l.989-1011). It is not used by the comparison
    pipeline, which uses the batched version below.
    """
    _require_torch()
    alphas = torch.linspace(0, 1, n_steps, dtype=baseline.dtype, device=baseline.device)
    diffs = target - baseline
    path_points = baseline + alphas[:, None] * diffs

    path_points.requires_grad_(True)
    outputs = model(path_points)
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=path_points,
        grad_outputs=torch.ones_like(outputs),
        create_graph=False,
    )[0]

    grad_feature = grads[:, feature_index].detach()
    integral = torch.trapz(grad_feature, alphas) * diffs[feature_index]
    return integral.item()


def _batched_line_integrals(model, base_points, diffs, n_steps: int = 50):
    """IG values for each (context, feature) where only that one feature changes.

    Output shape: (n_contexts, n_features).
    """
    _require_torch()
    n_contexts, n_features = base_points.shape
    alphas = torch.linspace(0, 1, n_steps, dtype=base_points.dtype, device=base_points.device)

    delta = torch.zeros((n_features, n_steps, n_features), dtype=base_points.dtype, device=base_points.device)
    idx = torch.arange(n_features, device=base_points.device)
    delta[idx, :, idx] = diffs[:, None] * alphas[None, :]

    path_points = base_points[:, None, None, :] + delta[None, :, :, :]
    flat = path_points.reshape(-1, n_features).requires_grad_(True)

    outputs = model(flat)
    if outputs.dim() > 1 and outputs.shape[-1] > 1:
        outputs = outputs[:, 0]
    outputs = outputs.squeeze(-1)

    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=flat,
        grad_outputs=torch.ones_like(outputs),
        create_graph=False,
    )[0]

    grads = grads.reshape(n_contexts, n_features, n_steps, n_features)
    grads_i = grads[:, idx, :, idx]
    integrals = torch.trapz(grads_i, alphas, dim=2) * diffs[None, :]
    return integrals


def ig_second_difference(model, x, baseline, n_steps: int = 50) -> np.ndarray:
    """A[i, j] = max(0, ig_first[i] - ig_second[j, i]) for one (x, baseline) pair.

    Formerly ``asymmetric_integrated_gradients`` (detect_claude.py l.1053-1071).
    """
    _require_torch()
    n_features = len(x)
    diffs = x - baseline

    ig_first = _batched_line_integrals(model, baseline.unsqueeze(0), diffs, n_steps).squeeze(0)

    base_contexts = baseline.repeat(n_features, 1)
    idx = torch.arange(n_features, device=baseline.device)
    base_contexts[idx, idx] = x[idx]
    ig_second = _batched_line_integrals(model, base_contexts, diffs, n_steps)

    A = torch.clamp(ig_first[None, :] - ig_second, min=0.0).transpose(0, 1)
    A.fill_diagonal_(0.0)
    return A.detach().cpu().numpy()


def compute_ig_second_difference_matrix(
    model,
    scaler_X,
    X_test: np.ndarray,
    X_train: np.ndarray,
    n_samples: int = 50,
    n_steps: int = 50,
    n_baselines: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Average :func:`ig_second_difference` over the first ``n_samples`` test rows
    and ``n_baselines`` random training rows (all standardised).

    Formerly ``compute_aig_matrix`` (detect_claude.py l.1074-1098).
    """
    _require_torch()
    rng = np.random.default_rng(seed)
    X_sc = scaler_X.transform(X_test[:n_samples])
    X_train_sc = scaler_X.transform(X_train)
    n_features = X_sc.shape[1]

    bg_idx = rng.choice(len(X_train_sc), size=min(n_baselines, len(X_train_sc)), replace=False)
    baselines = torch.tensor(X_train_sc[bg_idx], dtype=torch.float32)

    A_sum = np.zeros((n_features, n_features))
    n_evals = 0
    for row in X_sc:
        x = torch.tensor(row, dtype=torch.float32)
        for baseline in baselines:
            A_sum += ig_second_difference(model, x, baseline, n_steps)
            n_evals += 1
    return A_sum / n_evals


def second_difference_reference(f: Callable[[np.ndarray], float], x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Exact (quadrature-free) limit of :func:`ig_second_difference` for a NumPy function.

    Returns ``A[i, j] = max(0, -(f(b+d_i+d_j) - f(b+d_i) - f(b+d_j) + f(b)))`` with a
    zero diagonal. Added for testing and documentation; not in the original code.
    """
    x = np.asarray(x, dtype=float)
    b = np.asarray(baseline, dtype=float)
    n = len(x)
    d = x - b
    f_b = f(b)
    single = []
    for k in range(n):
        p = b.copy()
        p[k] += d[k]
        single.append(f(p))
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = b.copy()
            p[i] += d[i]
            p[j] += d[j]
            D = f(p) - single[i] - single[j] + f_b
            A[i, j] = max(0.0, -D)
    return A
