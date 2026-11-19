"""PC-algorithm DAG mask ("IG2 x PC").

Origin: ``detect_claude.py`` l.1196-1205 (``pc_directed_graph``) and l.1335
(``"IH+PC": IH_raw * PC_mat``). The original plots called this
"Integrated Hessians + PC".

What it computes
----------------
The PC algorithm (causal-learn) is run on the *feature matrix* ``X_train`` only;
it never sees the model or its predictions. Only edges that PC fully orients
(``i -> j``) become 1 in the mask; undirected edges are dropped. The "IG2 x PC"
matrix is the element-wise product of the (symmetric) IG2 matrix and this mask.

Consequently **all of its directionality comes from feature-level causal
discovery, not from the model**. For the default SCM, PC can orient the
v-structure X0 -> X2 <- X1 and (by Meek's rule) X2 -> X3, but it has no way to
orient X1 - X4, so that edge is dropped (consistent with the original figure
``results/original_detect_claude_2026-05-31/fig1_asymmetry_heatmaps.png``).
"""

from __future__ import annotations

import numpy as np


def directed_mask_from_causallearn_graph(mat: np.ndarray) -> np.ndarray:
    """Convert a causal-learn ``GeneralGraph.graph`` matrix into a binary directed mask.

    causal-learn encodes ``i -> j`` as ``mat[i, j] == -1 and mat[j, i] == 1``;
    an undirected edge is ``-1 / -1`` and is *not* included. Same loop as the
    original l.1199-1203, factored out so it can be tested without causal-learn.
    """
    mat = np.asarray(mat)
    n = mat.shape[0]
    directed = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if mat[i, j] == -1 and mat[j, i] == 1:
                directed[i, j] = 1.0
    return directed


def pc_directed_graph(X_data: np.ndarray, alpha: float = 0.05, ci_test: str = "kci") -> np.ndarray:
    """Run PC on ``X_data`` and return the directed-edge mask. Needs ``causal-learn``.

    Note: the comparison pipeline calls this with ``alpha=0.1, ci_test="fisherz"``
    (detect_claude.py l.1322), not with these defaults.
    """
    from causallearn.search.ConstraintBased.PC import pc  # lazy: optional dependency

    cg = pc(X_data, alpha=alpha, indep_test=ci_test, verbose=False)
    directed = directed_mask_from_causallearn_graph(cg.G.graph)
    print(f"PC discovered {int(np.sum(directed))} directed edges (unsupervised)")
    return directed


def apply_dag_mask(M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Element-wise product used for "IG2 x PC" (or, with the true DAG, an oracle mask)."""
    return np.asarray(M) * np.asarray(mask)
