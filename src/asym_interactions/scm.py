"""Synthetic 5-feature structural causal model (SCM) with a hand-set "ground truth".

Origin: ``detect_claude.py`` l.843-927 (``FEATURE_NAMES``, ``DEFAULT_CONFIG``,
``generate_asymmetric_dataset``). The data-generating process, the random-draw
order and the returned values are unchanged. The only refactors are:

* the noise-free target is computed by :func:`structural_target` (same formula,
  same position in the random stream), so tests can inspect it;
* the DAG edges and ground-truth weights are module constants
  (:data:`DAG_EDGES`, :data:`GT_WEIGHTS`) instead of literals inside the function.

Structure (default configuration)::

    X0 ~ N(0,1)          X1 ~ N(0,1)
    X2 = 0.6 X0 + 0.5 X1 + 0.3 e2         (0.6 tanh(2 X0) if causal_nonlinear)
    X3 = 0.7 X2 + 0.3 e3
    X4 = 0.5 X1 + 0.4 e4
    y  = X0 + 0.5 X1 + 5 X0 sin X2 + 4 X1 tanh X2 + 2 X2 X3 + 1.5 X1 X4 + noise

Important limitation (see README, "Methodological limitations"): the
ground-truth matrix ``GT`` places each interaction coefficient on the *causal
DAG edge* between the two features (``GT[0, 2] = 5`` but ``GT[2, 0] = 0``).
The interaction terms themselves are symmetric in the sense that their mixed
second differences are equal in both orders, so the asymmetry in ``GT`` is a
property of the feature DAG, not of the function ``y``. A method that recovers
the feature DAG scores well on this ground truth without reading the model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

FEATURE_NAMES = ["X0", "X1", "X2", "X3", "X4"]
N_FEATURES = len(FEATURE_NAMES)

DEFAULT_CONFIG = {
    "n_samples": 3000,
    "noise_level": 0.05,
    "interaction_type": "nonlinear",  # "nonlinear", "threshold", "polynomial"
    "include_confounding": False,
    "causal_nonlinear": False,
}

INTERACTION_TYPES = ("nonlinear", "threshold", "polynomial")

#: Directed edges of the feature-level causal DAG (parent, child).
DAG_EDGES = [(0, 2), (1, 2), (2, 3), (1, 4)]

#: Ground-truth "directional interaction" weights, placed on the DAG edges.
GT_WEIGHTS = {(0, 2): 5.0, (1, 2): 4.0, (2, 3): 2.0, (1, 4): 1.5}


def ground_truth_matrix(n_features: int = N_FEATURES) -> np.ndarray:
    """Return the 5x5 ground-truth matrix used for evaluation (GT[i, j] = influence i -> j)."""
    GT = np.zeros((n_features, n_features))
    for (i, j), w in GT_WEIGHTS.items():
        GT[i, j] = w
    return GT


def dag_adjacency(dag_edges=None, n_features: int = N_FEATURES) -> np.ndarray:
    """Binary adjacency matrix A[i, j] = 1 iff i -> j is a DAG edge."""
    if dag_edges is None:
        dag_edges = DAG_EDGES
    A = np.zeros((n_features, n_features))
    for i, j in dag_edges:
        A[i, j] = 1.0
    return A


def structural_target(X0, X1, X2, X3, X4, interaction_type: str = "nonlinear") -> np.ndarray:
    """Noise-free target y = f(X). Same formulas as detect_claude.py l.883-911."""
    if interaction_type == "nonlinear":
        return (
            1.0 * X0
            + 0.5 * X1
            + 5.0 * X0 * np.sin(X2)
            + 4.0 * X1 * np.tanh(X2)
            + 2.0 * X2 * X3
            + 1.5 * X1 * X4
        )
    if interaction_type == "threshold":
        return (
            1.0 * X0
            + 0.5 * X1
            + 5.0 * X0 * (X2 > 0).astype(float)
            + 4.0 * X1 * (X2 > 0).astype(float)
            + 2.0 * X2 * X3
            + 1.5 * X1 * X4
        )
    if interaction_type == "polynomial":
        return (
            1.0 * X0
            + 0.5 * X1
            + 5.0 * X0 * (X2 ** 2)
            + 4.0 * X1 * (X2 ** 2)
            + 2.0 * X2 * X3
            + 1.5 * X1 * X4
        )
    raise ValueError(f"Unknown interaction_type: {interaction_type}")


def generate_asymmetric_dataset(config: Optional[dict] = None, seed: Optional[int] = None):
    """Draw one dataset from the SCM.

    Returns ``(df, y, GT, dag_edges)``. Uses NumPy's *global* RNG
    (``np.random.seed``), exactly as the original script did.
    """
    if config is None:
        config = DEFAULT_CONFIG
    if seed is not None:
        np.random.seed(seed)

    n_samples = config["n_samples"]
    noise_level = config["noise_level"]
    interaction_type = config["interaction_type"]
    include_confounding = config.get("include_confounding", False)
    causal_nonlinear = config.get("causal_nonlinear", False)

    if include_confounding:
        U = np.random.randn(n_samples)
        X0 = 1.0 * U + 0.5 * np.random.randn(n_samples)
        X1 = 0.8 * U + 0.5 * np.random.randn(n_samples)
    else:
        X0 = np.random.randn(n_samples)
        X1 = np.random.randn(n_samples)

    if causal_nonlinear:
        X2 = 0.6 * np.tanh(2 * X0) + 0.5 * X1 + 0.3 * np.random.randn(n_samples)
    else:
        X2 = 0.6 * X0 + 0.5 * X1 + 0.3 * np.random.randn(n_samples)

    X3 = 0.7 * X2 + 0.3 * np.random.randn(n_samples)
    X4 = 0.5 * X1 + 0.4 * np.random.randn(n_samples)

    y = structural_target(X0, X1, X2, X3, X4, interaction_type)

    signal_var = np.var(y)
    noise = np.random.randn(n_samples) * np.sqrt(signal_var * noise_level)
    y = y + noise

    GT = ground_truth_matrix()
    dag_edges = list(DAG_EDGES)

    X = np.column_stack([X0, X1, X2, X3, X4])
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    return df, y, GT, dag_edges
