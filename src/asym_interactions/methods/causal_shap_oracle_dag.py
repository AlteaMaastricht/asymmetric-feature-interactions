"""DAG-ordered Shapley marginal-product matrix, given the TRUE DAG ("Causal SHAP").

Origin: ``detect_claude.py`` l.1101-1186 (``topological_orderings``,
``causal_shap_asymmetry``). The original plots called it "Causal SHAP". It is a
custom scheme, not the Causal Shapley values of Heskes et al. (2020) and not
the ``shap`` library. The computation is unchanged; only the names are honest.

What it computes
----------------
1. Orderings: (up to ``max_orderings``) topological orderings of the DAG passed
   in as ``dag_edges``. In the pipeline this is the **true DAG of the SCM**
   (oracle information; detect_claude.py l.1327). The default DAG has 7
   topological orderings, so all of them are used.
2. For each explained row ``x`` and each ordering, features are switched from
   the background distribution to ``x`` one by one (interventional /
   marginal replacement over a background sample, not conditional sampling).
   ``marginal[k]`` = change in the mean prediction when feature k is added.
3. For every pair where ``i`` precedes ``k`` in the ordering:
   ``M[i, k] += |marginal[k]| * |marginal[i]|``; the result is averaged over
   the number of times the pair was ordered that way.

Consequences:

* If ``i`` is an ancestor of ``k`` in the given DAG, ``i`` always precedes ``k``,
  so ``M[k, i] == 0`` exactly. The direction of every DAG edge is therefore
  **copied from the input DAG**; with the true DAG it is correct by construction.
* The entry is a product of two first-order marginal contributions. It is
  non-zero even for a purely additive model with **no interaction at all**.
"""

from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np


def topological_orderings(dag_edges, n_nodes, max_orderings=None, random_state: int = 42):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(dag_edges)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("dag_edges must define a DAG.")
    if max_orderings is None:
        return [list(order) for order in nx.all_topological_sorts(dag)]
    rng = np.random.default_rng(random_state)
    parents = {n: set(dag.predecessors(n)) for n in dag.nodes()}
    sampled = []
    seen = set()
    max_attempts = max_orderings * 20
    attempts = 0
    while len(sampled) < max_orderings and attempts < max_attempts:
        attempts += 1
        remaining = set(range(n_nodes))
        completed = set()
        order = []
        while remaining:
            available = [n for n in remaining if parents[n].issubset(completed)]
            if not available:
                break
            node = int(rng.choice(available))
            order.append(node)
            remaining.remove(node)
            completed.add(node)
        if len(order) == n_nodes:
            t = tuple(order)
            if t not in seen:
                seen.add(t)
                sampled.append(order)
    if not sampled:
        return [list(nx.topological_sort(dag))]
    return sampled


def dag_ordered_shapley_products(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_data: np.ndarray,
    dag_edges,
    feature_names=None,
    n_samples: int = 200,
    max_orderings: int = 100,
    background_size: int = 150,
    seed: int = 42,
) -> np.ndarray:
    """Formerly ``causal_shap_asymmetry``. ``feature_names`` is accepted but unused (as in the original)."""
    rng = np.random.default_rng(seed)
    n_features = X_data.shape[1]

    # Small background set
    bg_idx = rng.choice(len(X_data), size=min(background_size, len(X_data)), replace=False)
    X_background = X_data[bg_idx]

    # Rows to explain (drawn from the same X_data; may overlap the background)
    idx = rng.choice(len(X_data), size=min(n_samples, len(X_data)), replace=False)
    X_sub = X_data[idx]

    orderings = topological_orderings(dag_edges, n_features, max_orderings=max_orderings, random_state=seed)
    if len(orderings) == 0:
        orderings = [list(range(n_features))]
    asym_matrix = np.zeros((n_features, n_features))
    counts = np.zeros((n_features, n_features))

    baseline_pred = np.mean(predict_fn(X_background))

    for x in X_sub:
        for ordering in orderings:
            preds = {}
            coalition = []
            prev_pred = baseline_pred
            predecessors = []
            for feat in ordering:
                coalition.append(feat)
                X_pert = X_background.copy()
                for f in coalition:
                    X_pert[:, f] = x[f]
                curr_pred = np.mean(predict_fn(X_pert))
                marginal = curr_pred - prev_pred
                preds[feat] = marginal
                prev_pred = curr_pred
                for i in predecessors:
                    asym_matrix[i, feat] += abs(marginal) * abs(preds.get(i, 0))
                    counts[i, feat] += 1
                predecessors.append(feat)
    counts[counts == 0] = 1
    return asym_matrix / counts


# Backwards-compatible alias for readers comparing with the original script.
causal_shap_asymmetry = dag_ordered_shapley_products
