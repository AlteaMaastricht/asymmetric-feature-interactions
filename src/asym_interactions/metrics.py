"""Evaluation metrics comparing an interaction matrix ``M`` with the ground truth ``GT``.

Origin: ``detect_claude.py`` l.1189-1298, unchanged except that the matrix size
is read from the input (``M.shape[0]``) instead of the global ``N_FEATURES``
(same value, 5, in every experiment).

Caveats that matter when reading the numbers (also in the README):

* ``GT`` is supported exactly on the true DAG edges (see :mod:`.scm`), so the
  edge-recovery metrics (ROC-AUC, AUPRC, precision/recall/F1, top-k) and the
  direction metrics reward recovering the *feature DAG*. A binary DAG adjacency
  matrix with no model information scores perfectly on them.
* ``asymmetry_score`` is computed on the **raw, un-normalised** matrix, so it
  depends on each method's scale; values are not comparable across methods.
* ``directional_accuracy`` counts a tie (``M[i,j] == M[j,i]``) as wrong. For a
  (numerically) symmetric matrix the sign is decided by quadrature noise.
* ``top_k_accuracy`` uses k=5 while GT has only 4 non-zero entries, so its
  maximum is 0.8.
* ``rank_correlation`` clips negative Spearman correlations to 0.
* ``normalise`` is a min-max over the whole matrix including the diagonal.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def normalise(M: np.ndarray) -> np.ndarray:
    lo, hi = M.min(), M.max()
    if hi == lo:
        return np.zeros_like(M)
    return (M - lo) / (hi - lo)


def asymmetry_score(M: np.ndarray) -> float:
    N = M.shape[0]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    return float(np.mean([abs(M[i, j] - M[j, i]) for i, j in pairs]))


def frobenius_similarity(M: np.ndarray, GT: np.ndarray) -> float:
    Mn = normalise(M)
    GTn = normalise(GT)
    return 1.0 / (1.0 + np.linalg.norm(Mn - GTn, "fro"))


def directional_accuracy(M: np.ndarray, GT: np.ndarray) -> float:
    n = M.shape[0]
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if GT[i, j] != GT[j, i]:
                total += 1
                if (M[i, j] - M[j, i]) * (GT[i, j] - GT[j, i]) > 0:
                    correct += 1
    return correct / total if total > 0 else 1.0


def rank_correlation(M: np.ndarray, GT: np.ndarray) -> float:
    mask = ~np.eye(M.shape[0], dtype=bool)
    rho, _ = spearmanr(M[mask], GT[mask])
    return max(rho, 0.0)


def top_k_accuracy(M: np.ndarray, GT: np.ndarray, k: int = 5) -> float:
    mask = ~np.eye(M.shape[0], dtype=bool)
    flat = M[mask]
    flat_gt = GT[mask]
    top_indices = np.argpartition(-flat, k)[:k]
    return float(np.mean(flat_gt[top_indices] > 0))


def edge_recovery(M: np.ndarray, GT: np.ndarray, threshold: float = 0.15):
    Mn = normalise(M)
    GTn = normalise(GT)
    pred = (Mn > threshold).astype(int)
    true = (GTn > 0).astype(int)
    mask = ~np.eye(M.shape[0], dtype=bool)
    p, t = pred[mask], true[mask]
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def edge_recovery_auc(M: np.ndarray, GT: np.ndarray):
    Mn = normalise(M)
    GTn = normalise(GT)
    mask = ~np.eye(M.shape[0], dtype=bool)
    y_true = (GTn[mask] > 0).astype(int)
    y_scores = Mn[mask]
    if np.unique(y_true).size == 1:
        return 0.5, 0.5
    return roc_auc_score(y_true, y_scores), average_precision_score(y_true, y_scores)


def structural_hamming_distance(M: np.ndarray, GT: np.ndarray) -> float:
    Mn = normalise(M)
    GTn = normalise(GT)
    pred_dir = np.sign(Mn - Mn.T)
    true_dir = np.sign(GTn - GTn.T)
    mask = ~np.eye(M.shape[0], dtype=bool)
    diff = (pred_dir[mask] != true_dir[mask]).sum()
    return diff / mask.sum()


def all_metrics(M: np.ndarray, GT: np.ndarray, k: int = 5):
    prec, rec, f1 = edge_recovery(M, GT)
    roc, pr = edge_recovery_auc(M, GT)
    return {
        "asymmetry": asymmetry_score(M),
        "frobenius": frobenius_similarity(M, GT),
        "directional_acc": directional_accuracy(M, GT),
        "rank_corr": rank_correlation(M, GT),
        "top_k_acc": top_k_accuracy(M, GT, k=k),
        "hamming": structural_hamming_distance(M, GT),
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "roc_auc": roc,
        "auprc": pr,
    }


METRIC_KEYS = [
    "asymmetry", "frobenius", "directional_acc", "rank_corr", "top_k_acc",
    "hamming", "prec", "rec", "f1", "roc_auc", "auprc",
]
