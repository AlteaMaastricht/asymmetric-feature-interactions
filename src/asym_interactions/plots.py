"""Figures for the comparison experiment.

Origin: ``detect_claude.py`` l.834-841 (colours) and l.1392-1685 (plot functions).
Changes (all listed in docs/PROVENANCE.md):

* Method labels describe what is computed ("IG second-difference", "IG2 x PC
  mask", "DAG-ordered Shapley (oracle DAG)") instead of "Integrated Hessians",
  "Int. Hessians + PC", "Causal SHAP".
* Titles state the number of seeds; error bars are omitted when only one seed
  was run (the original drew zero-width / NaN error bars labelled "mean +- std").
* ``plot_directed_graphs``: title said "threshold=0.15" while the code used 0.10;
  the title now prints the threshold actually used.
* ``plot_diverging_heatmaps``: the subtitle said "blue = i->j, red = j->i", but
  with the ``RdBu_r`` colormap positive values (M[i,j] > M[j,i]) are red. The
  subtitle now matches the colours.
* The runtime plot labels the "IG2 x PC" bar as IG2 time + PC time (bug fix in
  ``experiments/run_compare.py``).
* ``seaborn`` is imported lazily, only by the three heatmap functions.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .metrics import normalise  # noqa: E402
from .scm import FEATURE_NAMES  # noqa: E402

#: Method keys used throughout the pipeline.
METHODS = ["IG2", "IG2xPC", "DAGShapOracle"]

METHOD_LABELS = {
    "GT": "Ground truth (DAG edges)",
    "IG2": "IG second-difference\n(symmetric)",
    "IG2xPC": "IG2 x PC DAG mask",
    "DAGShapOracle": "DAG-ordered Shapley\n(oracle DAG)",
}

COLORS = {
    "IG2": "#4C72B0",
    "IG2xPC": "#0066CC",
    "DAGShapOracle": "#DD8452",
    "GT": "#C44E52",
    "bg": "#F8F8F8",
    "grid": "#E0E0E0",
}


def _one_line(label: str) -> str:
    return label.replace("\n", " ")


def _seed_note(n_seeds: int) -> str:
    return f"n = {n_seeds} seed" + ("" if n_seeds == 1 else "s") + (
        " (single seed: no error bars)" if n_seeds == 1 else ", mean +- sample std"
    )


def _err(stds, n_seeds):
    return None if n_seeds < 2 else stds


def plot_heatmaps(matrices: dict, titles: list, filename: str):
    import seaborn as sns  # lazy

    n = len(matrices)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    fig.patch.set_facecolor(COLORS["bg"])

    for ax, (label, M), title in zip(axes, matrices.items(), titles):
        Mn = normalise(M)
        np.fill_diagonal(Mn, 0)
        color = COLORS.get(label, "gray")
        sns.heatmap(
            Mn, ax=ax, cmap=sns.light_palette(color, as_cmap=True),
            xticklabels=FEATURE_NAMES, yticklabels=FEATURE_NAMES,
            annot=True, fmt=".2f", linewidths=0.5, linecolor=COLORS["grid"],
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=8)
        ax.set_xlabel("Influenced (j)", fontsize=9)
        ax.set_ylabel("Influencer (i)", fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Interaction matrices (min-max normalised)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")


def plot_asymmetry_comparison(metric_means: dict, metric_stds: dict, filename: str, n_seeds: int = 1):
    methods = METHODS
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]
    asym_means = [metric_means[m]["asymmetry"] for m in methods]
    asym_stds = [metric_stds[m]["asymmetry"] for m in methods]
    frob_means = [metric_means[m]["frobenius"] for m in methods]
    frob_stds = [metric_stds[m]["frobenius"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax in axes:
        ax.set_facecolor(COLORS["bg"])
        ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

    x = np.arange(len(methods))
    w = 0.45
    axes[0].bar(x, asym_means, w, yerr=_err(asym_stds, n_seeds), capsize=5, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_title("Mean |M[i,j] - M[j,i]| on raw (un-normalised) matrices", fontsize=11)
    axes[0].set_ylabel("Score (method-specific scale)")

    axes[1].bar(x, frob_means, w, yerr=_err(frob_stds, n_seeds), capsize=5, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_title("Frobenius similarity to GT (normalised)", fontsize=11)
    axes[1].set_ylabel("Similarity")
    axes[1].set_ylim(0, 1.15)

    fig.suptitle(f"Method comparison ({_seed_note(n_seeds)})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")


def plot_edge_recovery_bar(metric_means: dict, metric_stds: dict, filename: str, n_seeds: int = 1):
    methods = METHODS
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]
    roc_means = [metric_means[m]["roc_auc"] for m in methods]
    roc_stds = [metric_stds[m]["roc_auc"] for m in methods]
    pr_means = [metric_means[m]["auprc"] for m in methods]
    pr_stds = [metric_stds[m]["auprc"] for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_facecolor(COLORS["bg"])
    ax.yaxis.grid(True, color=COLORS["grid"])
    ax.set_axisbelow(True)

    ax.bar(x - w / 2, roc_means, w, yerr=_err(roc_stds, n_seeds), capsize=5, color=colors, edgecolor="white", label="ROC-AUC")
    ax.bar(x + w / 2, pr_means, w, yerr=_err(pr_stds, n_seeds), capsize=5, color=[c + "99" for c in colors], edgecolor="white", label="AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Recovery of GT support (= true DAG edges); {_seed_note(n_seeds)}", fontsize=11)
    ax.set_ylabel("Score")
    ax.axhline(0.5, color=COLORS["GT"], linestyle="--", label="Random ROC")
    ax.legend()
    fig.suptitle("Threshold-free edge recovery", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")


def plot_roc_pr_curves(pooled_roc: dict, filename_roc: str, filename_pr: str):
    methods = METHODS
    labels = [_one_line(METHOD_LABELS[m]) for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(COLORS["bg"])
    for m, lbl, c in zip(methods, labels, colors):
        y_true, y_scores = pooled_roc[m]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lbl} (AUC={roc_auc_score(y_true, y_scores):.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC curves for GT support (pooled over seeds)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename_roc, dpi=150, facecolor=COLORS["bg"])
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(COLORS["bg"])
    for m, lbl, c in zip(methods, labels, colors):
        y_true, y_scores = pooled_roc[m]
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        ax.plot(rec_curve, prec_curve, color=c, lw=2, label=f"{lbl} (AUPRC={auprc:.2f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-recall curves for GT support (pooled over seeds)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename_pr, dpi=150, facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename_roc} and {filename_pr}")


def plot_directed_graphs(matrices: dict, filename: str, threshold: float = 0.10):
    method_order = ["GT"] + METHODS
    titles = [_one_line(METHOD_LABELS[k]) for k in method_order]
    colors = [COLORS[k] for k in method_order]
    n = len(FEATURE_NAMES)

    fig, axes = plt.subplots(1, len(method_order), figsize=(22, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    G_base = nx.DiGraph()
    G_base.add_nodes_from(range(n))
    pos = nx.circular_layout(G_base)

    for ax, key, title, col in zip(axes, method_order, titles, colors):
        ax.set_facecolor(COLORS["bg"])
        ax.set_title(title, fontsize=10, fontweight="bold", color=col)
        M = normalise(matrices[key].copy())
        np.fill_diagonal(M, 0)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if i != j and M[i, j] > threshold:
                    G.add_edge(i, j, weight=M[i, j])
        edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges()]
        edge_colors = [col] * len(G.edges())
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color=col, alpha=0.85)
        nx.draw_networkx_labels(G, pos, {i: FEATURE_NAMES[i] for i in range(n)}, ax=ax, font_color="white", font_size=9, font_weight="bold")
        if G.edges():
            nx.draw_networkx_edges(
                G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, arrows=True,
                arrowsize=18, connectionstyle="arc3,rad=0.1", min_source_margin=20, min_target_margin=20,
            )
        ax.axis("off")
    fig.suptitle(f"Thresholded interaction graphs (normalised entry > {threshold:.2f})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")


def plot_diverging_heatmaps(matrices: dict, filename: str):
    import seaborn as sns  # lazy

    method_order = ["GT"] + METHODS
    titles = [_one_line(METHOD_LABELS[k]) for k in method_order]
    n = len(method_order)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor(COLORS["bg"])
    for ax, key, title in zip(axes, method_order, titles):
        M = normalise(matrices[key].copy())
        D = M - M.T
        vmax = np.max(np.abs(D)) if np.max(np.abs(D)) > 0 else 1
        sns.heatmap(
            D, ax=ax, cmap="RdBu_r", center=0, xticklabels=FEATURE_NAMES, yticklabels=FEATURE_NAMES,
            annot=True, fmt=".2f", linewidths=0.5, linecolor=COLORS["grid"], vmin=-vmax, vmax=vmax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{title}\n(M[i,j] - M[j,i]; colour scale per panel)", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=9)
    fig.suptitle("Antisymmetric part (red = M[i,j] > M[j,i], blue = M[i,j] < M[j,i])", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")


def plot_rank_consistency(matrices: dict, filename: str):
    import seaborn as sns  # lazy

    methods = METHODS
    labels = methods
    n = len(methods)
    corr = np.eye(n)
    mask = ~np.eye(len(FEATURE_NAMES), dtype=bool)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                rho, _ = spearmanr(matrices[m1][mask], matrices[m2][mask])
                corr[i, j] = corr[j, i] = rho
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    ax.set_title("Spearman rank correlation between methods (off-diagonal entries)", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")


def plot_radar_chart(metric_means: dict, filename: str, n_seeds: int = 1):
    methods = METHODS
    labels = [_one_line(METHOD_LABELS[m]) for m in methods]
    colors = [COLORS[m] for m in methods]
    metric_keys = ["asymmetry", "directional_acc", "rank_corr", "frobenius", "roc_auc", "auprc"]
    metric_names = ["Asymmetry (raw scale)", "Direc. Acc.", "Rank Corr.", "Frob. Sim.", "ROC-AUC", "AUPRC"]
    n_metrics = len(metric_keys)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(COLORS["bg"])
    for m, lbl, c in zip(methods, labels, colors):
        values = [metric_means[m][k] for k in metric_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", color=c, linewidth=2, label=lbl)
        ax.fill(angles, values, alpha=0.1, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_title(f"Multi-metric comparison ({_seed_note(n_seeds)})", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")


def plot_runtime_complexity(runtime_means: list, runtime_stds: list, filename: str, n_seeds: int = 1, budgets: str = ""):
    methods = METHODS
    labels = ["IG2", "IG2 + PC\n(IG2 time + PC time)", "DAG-ordered Shapley\n(oracle DAG)"]
    colors = [COLORS[m] for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(COLORS["bg"])
    ax.yaxis.grid(True, color=COLORS["grid"])
    ax.set_axisbelow(True)
    bars = ax.bar(labels, runtime_means, yerr=_err(runtime_stds, n_seeds), capsize=5, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, runtime_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{val:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"Wall-clock runtime ({_seed_note(n_seeds)})", fontsize=12, fontweight="bold")
    if budgets:
        ax.text(0.01, 0.97, budgets, transform=ax.transAxes, fontsize=8, va="top")
    ax.set_ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved -> {filename}")
