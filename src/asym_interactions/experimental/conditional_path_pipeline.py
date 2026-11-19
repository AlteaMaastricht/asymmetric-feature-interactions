"""
================================================================================
EXPERIMENTAL / UNUSED: conditional-integration-path attribution pipeline
================================================================================
Copied verbatim from ``detect_claude.py`` l.15-829 (imports, data classes,
``CausalDiscovery``, ``ConditionalIntegrationPaths``,
``AsymmetricIntegratedGradients``, ``CausalXAIPipeline``, demo). Only this
header was added.

Status -- read before using:

* **None of the thesis figures were produced by this code.** The comparison
  pipeline (``run_comparative_analysis`` in the original, now
  ``asym_interactions.experiments.run_compare``) never calls it. It is kept
  because it is the only place where the thesis idea of *conditional*
  integration paths (Nadaraya-Watson conditional means along a DAG order) is
  implemented.
* It produces **per-feature** attributions (a vector), weighted by the number of
  causal descendants of each feature. It does not produce a pairwise
  interaction matrix and was never evaluated against the ground truth.
* Untested. Needs torch, causal-learn, seaborn and pandas at import time.
* Known issue: ``CausalDiscovery._detect_latent_confounders`` treats a
  causal-learn ``-1/-1`` entry as a bidirected (latent-confounder) edge. For PC
  output ``-1/-1`` means an *undirected* edge (bidirected edges are ``1/1`` in
  FCI output), so FCI is triggered whenever PC leaves an edge unoriented.
* Known issue: ``demo_pipeline`` builds its label from ``Smoking`` and ``BMI``,
  not from the ``risk`` variable that ``generate_synthetic_scm`` computes and
  then discards.
================================================================================
"""
from __future__ import annotations

import warnings
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from scipy.stats import norm, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader, TensorDataset

# causal-learn imports
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, chisq, kci

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CausalXAI")


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalGraph:
    """Immutable result produced by Phase 1."""
    dag:            nx.DiGraph          # Directed Acyclic Graph
    adjacency:      np.ndarray          # Raw adjacency matrix (n_features × n_features)
    feature_names:  list[str]
    algorithm:      str                 # "PC" or "FCI"
    has_latent_confounders: bool = False
    # FCI-specific: bidirected edges (potential latent confounders)
    bidirected_edges: list[tuple] = field(default_factory=list)

    def ancestors_of(self, node: int) -> list[int]:
        return list(nx.ancestors(self.dag, node))

    def causal_order(self) -> list[int]:
        return list(nx.topological_sort(self.dag))

    def markov_blanket(self, node: int) -> list[int]:
        parents  = list(self.dag.predecessors(node))
        children = list(self.dag.successors(node))
        spouses  = [
            p for c in children
            for p in self.dag.predecessors(c)
            if p != node
        ]
        return list(set(parents + children + spouses))


@dataclass
class IntegrationPath:
    """A single non-linear integration curve γ(α) for one sample."""
    alpha_steps:    np.ndarray   # shape (n_steps,)          monotone in [0,1]
    waypoints:      np.ndarray   # shape (n_steps, n_features) points along γ
    feature_names:  list[str]
    baseline:       np.ndarray   # x'
    explicand:      np.ndarray   # x


@dataclass
class AttributionResult:
    """Output of Phase 3."""
    attributions:       np.ndarray      # shape (n_features,)
    asymmetry_weights:  np.ndarray      # shape (n_features,)  causal strength
    weighted_attrs:     np.ndarray      # attributions × asymmetry_weights
    prediction:         float
    baseline_prediction: float
    completeness_gap:   float           # |Σ attr - (f(x)-f(x'))|  ≈ 0 if correct
    feature_names:      list[str]

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        idx = np.argsort(np.abs(self.weighted_attrs))[::-1][:k]
        return [(self.feature_names[i], float(self.weighted_attrs[i])) for i in idx]

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Prediction        : {self.prediction:.4f}",
            f"  Baseline pred.    : {self.baseline_prediction:.4f}",
            f"  Δ (f(x)−f(x'))    : {self.prediction - self.baseline_prediction:.4f}",
            f"  Completeness gap  : {self.completeness_gap:.6f}",
            f"\n  Top feature attributions (causal-weighted):",
        ]
        for name, val in self.top_k(k=min(8, len(self.feature_names))):
            bar = "█" * int(abs(val) * 40 / (max(abs(self.weighted_attrs)) + 1e-9))
            sign = "+" if val >= 0 else "-"
            lines.append(f"    {sign} {name:<25} {val:+.4f}  {bar}")
        lines.append("─"*60)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Unsupervised Structural Causal Discovery
# ──────────────────────────────────────────────────────────────────────────────

class CausalDiscovery:
    """
    Runs the PC algorithm (or FCI if latent confounders are detected) to
    extract a Directed Acyclic Graph from raw observational data.

    Parameters
    ----------
    alpha           : significance threshold for conditional independence tests
    indep_test      : 'fisherz' (continuous), 'chisq' (discrete), 'kci' (nonlinear)
    max_cond_set    : max conditioning-set size (None = unlimited)
    stable          : use stable-PC (order-independent skeleton)
    fci_fallback    : automatically switch to FCI when PAG contains bidirected edges
    """

    def __init__(
        self,
        alpha:          float = 0.05,
        indep_test:     str   = "fisherz",
        max_cond_set:   Optional[int] = None,
        stable:         bool  = True,
        fci_fallback:   bool  = True,
    ):
        self.alpha        = alpha
        self.indep_test   = indep_test
        self.max_cond_set = max_cond_set
        self.stable       = stable
        self.fci_fallback = fci_fallback

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> CausalGraph:
        """
        Parameters
        ----------
        X             : raw data matrix, shape (n_samples, n_features)
        feature_names : optional human-readable column names
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]

        log.info("Phase 1 ▶ Running PC algorithm  (α=%.3f, test=%s)", self.alpha, self.indep_test)

        # ── PC algorithm ────────────────────────────────────────────────
        cg = pc(
            data           = X,
            alpha          = self.alpha,
            indep_test     = self.indep_test,
            stable         = self.stable,
            uc_rule        = 0,
            uc_priority    = 2,
            mvpc           = False,
            correction_name= None,
            background_knowledge = None,
            verbose        = False,
            show_progress  = False,
        )

        adj_matrix = cg.G.graph          # shape (n, n); ±1 encoding

        has_latent, bidirected = self._detect_latent_confounders(adj_matrix)
        algorithm = "PC"

        if has_latent and self.fci_fallback:
            log.info("Phase 1 ▶ Bidirected edges detected — switching to FCI algorithm")
            adj_matrix, bidirected = self._run_fci(X)
            algorithm = "FCI"

        dag = self._build_dag(adj_matrix, feature_names)

        graph = CausalGraph(
            dag                    = dag,
            adjacency              = adj_matrix,
            feature_names          = feature_names,
            algorithm              = algorithm,
            has_latent_confounders = has_latent,
            bidirected_edges       = bidirected,
        )

        n_edges = dag.number_of_edges()
        log.info("Phase 1 ✔ Causal graph: %d nodes, %d directed edges  [%s]",
                 n_features, n_edges, algorithm)
        return graph

    # ------------------------------------------------------------------ #
    def _detect_latent_confounders(self, adj: np.ndarray) -> tuple[bool, list]:
        """
        In causal-learn's encoding:
          adj[i,j]=-1, adj[j,i]= 1  →  i → j
          adj[i,j]=-1, adj[j,i]=-1  →  i ↔ j  (bidirected / latent confounder)
        """
        n = adj.shape[0]
        bidirected = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == -1 and adj[j, i] == -1:
                    bidirected.append((i, j))
        return len(bidirected) > 0, bidirected

    def _run_fci(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        """Run FCI and return its PAG adjacency + bidirected edges."""
        g, edges = fci(
            dataset    = X,
            independence_test_method = self.indep_test,
            alpha      = self.alpha,
            verbose    = False,
            show_progress = False,
        )
        adj = g.graph
        _, bidirected = self._detect_latent_confounders(adj)
        return adj, bidirected

    def _build_dag(self, adj: np.ndarray, names: list[str]) -> nx.DiGraph:
        """Convert causal-learn adjacency matrix to a NetworkX DAG."""
        G = nx.DiGraph()
        n = adj.shape[0]
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(n):
                # adj[i,j]=-1 and adj[j,i]=1  ⟹  i → j
                if adj[i, j] == -1 and adj[j, i] == 1:
                    G.add_edge(i, j)

        # Remove cycles introduced by undirected/ambiguous edges
        if not nx.is_directed_acyclic_graph(G):
            log.warning("Phase 1 ⚠  Cycle detected — breaking with minimum-feedback-arc heuristic")
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                if len(cycle) >= 2:
                    G.remove_edge(cycle[-1], cycle[0])

        nx.set_node_attributes(G, {i: names[i] for i in range(n)}, "label")
        return G


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Manifold Mapping & Conditional Integration Path Parametrisation
# ──────────────────────────────────────────────────────────────────────────────

class ConditionalIntegrationPaths:
    """
    Constructs a non-linear integration curve γ(α) : [0,1] → ℝ^d that:
      • starts at the baseline x'
      • ends at the explicand x
      • at each α, respects the conditional distributions P(X_i | Pa(X_i))
        imposed by the causal DAG, keeping every waypoint on the data manifold.

    Strategy
    --------
    1. Derive the causal topological order from the DAG.
    2. For each feature i in causal order, compute a *conditional drift*
       δ_i(α) = E[X_i | Pa(X_i) = pa_i(α)] − x'_i,
       where pa_i(α) are the parent values at step α (already updated).
    3. The path waypoint at step α is:
           γ_i(α) = x'_i + α · δ_i(α)  +  ε_i(α)
       where ε_i(α) is a small Gaussian noise ~ N(0, σ²_i(1−α)) that
       vanishes at α=1, ensuring γ(1)=x exactly.
    4. Asymmetry weighting: features with many causal descendants receive
       higher weight in the attribution phase (computed here, passed to AIG).

    Parameters
    ----------
    n_steps         : number of quadrature points along γ
    kernel_bw       : bandwidth for Nadaraya-Watson conditional mean estimator
    noise_std_scale : scale of manifold-preserving noise (0 = straight line)
    """

    def __init__(
        self,
        n_steps:         int   = 50,
        kernel_bw:       float = 1.0,
        noise_std_scale: float = 0.05,
    ):
        self.n_steps         = n_steps
        self.kernel_bw       = kernel_bw
        self.noise_std_scale = noise_std_scale

    # ------------------------------------------------------------------ #
    def build_paths(
        self,
        X_train:  np.ndarray,
        causal_graph: CausalGraph,
        baseline: np.ndarray,       # x'  shape (n_features,)
        explicands: np.ndarray,     # X   shape (n_samples, n_features)
    ) -> tuple[list[IntegrationPath], np.ndarray]:
        """
        Returns
        -------
        paths             : one IntegrationPath per explicand sample
        asymmetry_weights : shape (n_features,) — causal descendant count normalised
        """
        log.info("Phase 2 ▶ Building conditional integration paths  (steps=%d)", self.n_steps)

        causal_order = causal_graph.causal_order()
        asym_weights = self._compute_asymmetry_weights(causal_graph)

        # Per-feature conditional std from training data
        feature_std = np.std(X_train, axis=0) + 1e-8

        alpha_steps = np.linspace(0.0, 1.0, self.n_steps)
        paths = []

        for sample_idx, x in enumerate(explicands):
            waypoints = self._parametrise_path(
                X_train       = X_train,
                causal_graph  = causal_graph,
                causal_order  = causal_order,
                baseline      = baseline,
                explicand     = x,
                alpha_steps   = alpha_steps,
                feature_std   = feature_std,
            )
            paths.append(IntegrationPath(
                alpha_steps   = alpha_steps,
                waypoints     = waypoints,
                feature_names = causal_graph.feature_names,
                baseline      = baseline.copy(),
                explicand     = x.copy(),
            ))

        log.info("Phase 2 ✔ %d integration paths constructed", len(paths))
        return paths, asym_weights

    # ------------------------------------------------------------------ #
    def _parametrise_path(
        self,
        X_train:      np.ndarray,
        causal_graph: CausalGraph,
        causal_order: list[int],
        baseline:     np.ndarray,
        explicand:    np.ndarray,
        alpha_steps:  np.ndarray,
        feature_std:  np.ndarray,
    ) -> np.ndarray:
        """
        Core path builder.  Returns waypoints of shape (n_steps, n_features).
        """
        n_features = len(baseline)
        n_steps = len(alpha_steps)
        waypoints = np.tile(baseline, (n_steps, 1))
        one_minus_alpha = 1.0 - alpha_steps

        # Traverse features in causal order so parents are updated first
        for feat_idx in causal_order:
            parents = list(causal_graph.dag.predecessors(feat_idx))

            if len(parents) == 0:
                # Root node — simple linear interpolation
                cond_mean = baseline[feat_idx] + alpha_steps * (
                    explicand[feat_idx] - baseline[feat_idx]
                )
            else:
                # Conditional mean via batched Nadaraya-Watson on training data
                cond_mean = self._nadaraya_watson_batch(
                    X_train      = X_train,
                    query_parent = waypoints[:, parents],
                    parent_cols  = parents,
                    target_col   = feat_idx,
                )
                # Blend baseline → conditional mean → explicand
                cond_mean = (
                    baseline[feat_idx]
                    + alpha_steps * (explicand[feat_idx] - baseline[feat_idx])
                    + alpha_steps * one_minus_alpha * (cond_mean - baseline[feat_idx])
                )

            # Decaying noise that vanishes at α=1, preserving γ(1)=x
            noise_std = self.noise_std_scale * feature_std[feat_idx] * one_minus_alpha
            noise = np.random.normal(0.0, noise_std) if np.max(noise_std) > 1e-10 else 0.0

            gamma_col = cond_mean + noise
            # Hard-clamp at α=1 to guarantee endpoint exactness
            gamma_col[-1] = explicand[feat_idx]
            waypoints[:, feat_idx] = gamma_col

        return waypoints

    def _nadaraya_watson(
        self,
        X_train:      np.ndarray,
        query_parent: np.ndarray,
        parent_cols:  list[int],
        target_col:   int,
    ) -> float:
        """Kernel regression E[X_target | X_parents = query]."""
        parent_data = X_train[:, parent_cols]
        diffs       = parent_data - query_parent          # (n, len(parents))
        sq_dist     = np.sum(diffs ** 2, axis=1)
        weights     = np.exp(-sq_dist / (2 * self.kernel_bw ** 2))
        w_sum       = weights.sum()
        if w_sum < 1e-12:
            return float(X_train[:, target_col].mean())
        return float(np.dot(weights, X_train[:, target_col]) / w_sum)

    def _nadaraya_watson_batch(
        self,
        X_train:      np.ndarray,
        query_parent: np.ndarray,
        parent_cols:  list[int],
        target_col:   int,
    ) -> np.ndarray:
        """Kernel regression E[X_target | X_parents = query] for a batch of queries."""
        parent_data = X_train[:, parent_cols]  # (n_train, n_parents)
        diffs = parent_data[None, :, :] - query_parent[:, None, :]  # (n_steps, n_train, n_parents)
        sq_dist = np.sum(diffs ** 2, axis=2)
        weights = np.exp(-sq_dist / (2 * self.kernel_bw ** 2))
        w_sum = weights.sum(axis=1)
        target_vals = X_train[:, target_col]
        weighted = weights @ target_vals
        denom = np.where(w_sum < 1e-12, 1.0, w_sum)
        cond_mean = weighted / denom
        cond_mean[w_sum < 1e-12] = target_vals.mean()
        return cond_mean

    def _compute_asymmetry_weights(self, causal_graph: CausalGraph) -> np.ndarray:
        """
        Asymmetry weight for feature i  =  (1 + |descendants(i)|) / (1 + max_desc).
        Features that are causal ancestors of many other features carry more weight.
        """
        n = len(causal_graph.feature_names)
        desc_counts = np.array([
            len(nx.descendants(causal_graph.dag, i)) for i in range(n)
        ], dtype=float)
        max_desc = desc_counts.max() if desc_counts.max() > 0 else 1.0
        weights  = (1.0 + desc_counts) / (1.0 + max_desc)
        log.info("Phase 2 ▷ Asymmetry weights (top-3 causal): %s",
                 sorted(zip(causal_graph.feature_names, weights),
                        key=lambda t: -t[1])[:3])
        return weights


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Asymmetric Integrated Gradients (AIG)
# ──────────────────────────────────────────────────────────────────────────────

class AsymmetricIntegratedGradients:
    """
    Computes feature attributions by integrating the model's gradients
    along the non-linear conditional paths γ(α) supplied by Phase 2.

    The standard IG formula:

        IG_i(x) = (x_i − x'_i) ∫₀¹ ∂f(γ(α))/∂x_i  dα

    is extended here to:

        AIG_i(x) = w_i · (x_i − x'_i) ∫₀¹ ∂f(γ(α))/∂x_i  dα

    where w_i is the causal asymmetry weight from Phase 2 (proportional
    to the number of causal descendants of feature i), and the integral
    is evaluated along the curved manifold path rather than a straight line.

    The trapezoidal rule is applied over the discrete waypoints.

    Parameters
    ----------
    model           : a callable PyTorch nn.Module, output shape (batch,) or (batch,1)
    target_class    : for multi-output models, which output index to explain
    device          : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model:        nn.Module,
        target_class: int = 0,
        device:       str = "cpu",
    ):
        self.model        = model.to(device)
        self.model.eval()
        self.target_class = target_class
        self.device       = device

    # ------------------------------------------------------------------ #
    def attribute(
        self,
        path:             IntegrationPath,
        asymmetry_weights: np.ndarray,
    ) -> AttributionResult:
        """
        Compute AIG attributions for a single sample.

        Parameters
        ----------
        path              : IntegrationPath from Phase 2
        asymmetry_weights : (n_features,) array from Phase 2
        """
        n_steps, n_features = path.waypoints.shape

        # ── Accumulate ∂f/∂x_i at each waypoint via batched autograd ──
        wp = torch.tensor(
            path.waypoints,
            dtype  = torch.float32,
            device = self.device,
        ).requires_grad_(True)
        outputs = self._forward(wp)
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=wp,
            grad_outputs=torch.ones_like(outputs),
            create_graph=False,
        )[0].detach().cpu().numpy()

        # ── Trapezoidal quadrature along γ(α) ─────────────────────────
        # dγ/dα ≈ finite difference of waypoints
        d_gamma = np.diff(path.waypoints, axis=0)          # (n_steps-1, n_features)
        g_mid   = 0.5 * (grads[:-1] + grads[1:])           # midpoint gradients

        # Path integral: Σ_k  ∇f(γ(αk)) · Δγ_k
        raw_attributions = np.sum(g_mid * d_gamma, axis=0)  # (n_features,)

        # ── Apply asymmetric causal weighting ─────────────────────────
        weighted_attrs = asymmetry_weights * raw_attributions

        # ── Predictions ───────────────────────────────────────────────
        with torch.no_grad():
            pred = float(self._forward(
                torch.tensor(path.explicand, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            ).item())
            base_pred = float(self._forward(
                torch.tensor(path.baseline, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            ).item())

        # Completeness: Σ raw_attrs ≈ f(x) − f(x')
        completeness_gap = abs(raw_attributions.sum() - (pred - base_pred))

        return AttributionResult(
            attributions        = raw_attributions,
            asymmetry_weights   = asymmetry_weights,
            weighted_attrs      = weighted_attrs,
            prediction          = pred,
            baseline_prediction = base_pred,
            completeness_gap    = completeness_gap,
            feature_names       = path.feature_names,
        )

    def attribute_batch(
        self,
        paths:             list[IntegrationPath],
        asymmetry_weights: np.ndarray,
    ) -> list[AttributionResult]:
        """Attribute a list of samples."""
        results = []
        for i, path in enumerate(paths):
            log.info("Phase 3 ▷ Attributing sample %d / %d", i + 1, len(paths))
            results.append(self.attribute(path, asymmetry_weights))
        return results

    # ------------------------------------------------------------------ #
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.dim() > 1 and out.shape[-1] > 1:
            out = out[:, self.target_class]
        return out.squeeze(-1).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Master Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class CausalXAIPipeline:
    """
    Fully automated end-to-end pipeline:

        raw data → causal DAG → manifold paths → AIG attributions

    Usage
    -----
    pipeline = CausalXAIPipeline(model=my_cnn, feature_names=[...])
    results  = pipeline.run(X_train, X_explain, baseline=X_mean)
    for r in results:
        print(r.summary())
    """

    def __init__(
        self,
        model:          nn.Module,
        feature_names:  Optional[list[str]]  = None,
        # Phase 1 config
        alpha:          float = 0.05,
        indep_test:     str   = "fisherz",
        fci_fallback:   bool  = True,
        # Phase 2 config
        n_steps:        int   = 50,
        kernel_bw:      float = 1.0,
        noise_std_scale:float = 0.05,
        # Phase 3 config
        target_class:   int   = 0,
        device:         str   = "cpu",
    ):
        self.feature_names = feature_names
        self.scaler        = StandardScaler()

        self.discovery = CausalDiscovery(
            alpha       = alpha,
            indep_test  = indep_test,
            fci_fallback= fci_fallback,
        )
        self.path_builder = ConditionalIntegrationPaths(
            n_steps         = n_steps,
            kernel_bw       = kernel_bw,
            noise_std_scale = noise_std_scale,
        )
        self.aig = AsymmetricIntegratedGradients(
            model        = model,
            target_class = target_class,
            device       = device,
        )

        # Stored after fit()
        self.causal_graph: Optional[CausalGraph] = None
        self.X_train_scaled: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def fit(self, X_train: np.ndarray) -> "CausalXAIPipeline":
        """
        Phase 1: learn the causal structure from training data.
        Scales data internally (z-score) for stable CI testing.
        """
        log.info("═" * 60)
        log.info("CausalXAI Pipeline — fit()")
        log.info("═" * 60)

        X_scaled = self.scaler.fit_transform(X_train)
        self.X_train_scaled = X_scaled

        self.causal_graph = self.discovery.fit(X_scaled, self.feature_names)
        return self

    def explain(
        self,
        X_explain:  np.ndarray,
        baseline:   Optional[np.ndarray] = None,
    ) -> list[AttributionResult]:
        """
        Phases 2 & 3: build paths and compute AIG attributions.

        Parameters
        ----------
        X_explain : samples to explain, shape (n_samples, n_features)
        baseline  : reference point x'; defaults to training mean (all-zeros in scaled space)
        """
        if self.causal_graph is None or self.X_train_scaled is None:
            raise RuntimeError("Call pipeline.fit(X_train) before pipeline.explain().")

        log.info("═" * 60)
        log.info("CausalXAI Pipeline — explain()  (%d samples)", len(X_explain))
        log.info("═" * 60)

        X_explain_scaled = self.scaler.transform(X_explain)

        if baseline is None:
            baseline_scaled = np.zeros(X_explain_scaled.shape[1])  # mean in scaled space
        else:
            baseline_scaled = self.scaler.transform(baseline.reshape(1, -1)).squeeze()

        # Phase 2
        paths, asym_weights = self.path_builder.build_paths(
            X_train      = self.X_train_scaled,
            causal_graph = self.causal_graph,
            baseline     = baseline_scaled,
            explicands   = X_explain_scaled,
        )

        # Phase 3
        log.info("Phase 3 ▶ Asymmetric Integrated Gradients attribution")
        results = self.aig.attribute_batch(paths, asym_weights)
        log.info("Phase 3 ✔ Attribution complete")
        return results

    def run(
        self,
        X_train:   np.ndarray,
        X_explain: np.ndarray,
        baseline:  Optional[np.ndarray] = None,
    ) -> list[AttributionResult]:
        """Convenience: fit + explain in one call."""
        return self.fit(X_train).explain(X_explain, baseline)


# ──────────────────────────────────────────────────────────────────────────────
# Demo — Synthetic Tabular Classifier
# ──────────────────────────────────────────────────────────────────────────────

class _DemoClassifier(nn.Module):
    """Small MLP for demonstration purposes."""
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_synthetic_scm(n_samples: int = 400, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """
    Synthetic Structural Causal Model (medical-flavoured):

        Age  →  BMI  →  Glucose  →  Risk
        Age  →  BP   →  Risk
        BMI  →  BP
        Smoking  →  Risk
    """
    rng = np.random.default_rng(seed)
    n   = n_samples

    age     = rng.normal(50,  10,  n)
    smoking = rng.binomial(1,  0.3, n).astype(float)
    bmi     = 0.15 * age + rng.normal(25, 3, n)
    bp      = 0.10 * age + 0.20 * bmi + rng.normal(120, 10, n)
    glucose = 0.30 * bmi + rng.normal(100, 15, n)
    risk    = (
        0.02 * age + 0.30 * smoking + 0.015 * bmi
        + 0.01 * bp + 0.008 * glucose
        + rng.normal(0, 0.5, n)
    )
    risk = 1 / (1 + np.exp(-0.1 * (risk - risk.mean())))  # sigmoid scale

    X     = np.stack([age, bmi, bp, glucose, smoking], axis=1)
    names = ["Age", "BMI", "BloodPressure", "Glucose", "Smoking"]
    return X, names


def demo_pipeline():
    log.info("Generating synthetic SCM dataset …")
    np.random.seed(0)
    torch.manual_seed(0)

    X, names = generate_synthetic_scm(n_samples=300)

    # Toy label: risk > 0.5
    scaler_y = StandardScaler()
    y = (X[:, -1] + 0.3 * X[:, 1] > X[:, -1].mean()).astype(np.float32)

    # Train a quick MLP
    model = _DemoClassifier(n_features=len(names))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    X_t = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    log.info("Training demo MLP …")
    model.train()
    for epoch in range(80):
        optimiser.zero_grad()
        loss = criterion(model(X_t).squeeze(), y_t)
        loss.backward()
        optimiser.step()
    model.eval()
    log.info("MLP training done  (final BCE loss: %.4f)", loss.item())

    # ── Run the full pipeline ────────────────────────────────────────────
    pipeline = CausalXAIPipeline(
        model          = model,
        feature_names  = names,
        alpha          = 0.05,
        indep_test     = "fisherz",
        fci_fallback   = True,
        n_steps        = 40,
        kernel_bw      = 0.8,
        noise_std_scale= 0.04,
        target_class   = 0,
        device         = "cpu",
    )

    X_train   = X[:250]
    X_explain = X[250:253]   # explain three samples

    results = pipeline.run(X_train, X_explain)

    print("\n" + "═" * 60)
    print("  CAUSAL XAI — ATTRIBUTION RESULTS")
    print("═" * 60)

    for i, res in enumerate(results):
        print(f"\n  Sample {i + 1}:")
        print(res.summary())

    print("\nCausal Graph edges:")
    for src, dst in pipeline.causal_graph.dag.edges():
        print(f"  {names[src]}  →  {names[dst]}")

    return pipeline, results

