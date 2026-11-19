"""Comparison experiment: IG2 vs IG2 x PC mask vs DAG-ordered Shapley (oracle DAG).

Origin: ``detect_claude.py`` l.1301-1389 (``run_single_seed``,
``run_multi_seed_evaluation``) and l.1688-1734 (``run_comparative_analysis``).
Defaults reproduce the original settings. Changes (see docs/PROVENANCE.md):

1. ``--seeds`` and the other CLI arguments (the original hard-coded ``seeds = [42]``).
2. Relative output directory (``--outdir``, default ``outputs_compare``).
3. **Bug fix:** the "IG2 x PC" runtime is IG2 time + PC time. The original
   reported PC time only (l.1338 ``runtimes = [t_ih_raw, t_pc, t_cs]``).
4. **Bug fix:** standard deviations use ``np.std(..., ddof=1)`` for both metrics
   and runtimes when >= 2 seeds, and NaN for a single seed. The original used a
   population std for metrics (0 for one seed) and ``ddof=1`` for runtimes
   (NaN for one seed), while the plots claimed "mean +- std over seeds".
5. Numeric outputs are written to CSV/NPZ/JSON (the original saved figures only).
   The test-set R^2 of the explained MLP is recorded (it was never measured).
6. The representative matrices for the figures are those of the first seed's
   run, instead of re-running that seed a second time (same code, same seed).

Run (needs torch, causal-learn, seaborn)::

    PYTHONPATH=src python -m asym_interactions.experiments.run_compare --seeds 42
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np

from ..metrics import METRIC_KEYS, all_metrics, normalise
from ..scm import DEFAULT_CONFIG, FEATURE_NAMES, INTERACTION_TYPES, generate_asymmetric_dataset

METHODS = ["IG2", "IG2xPC", "DAGShapOracle"]

DEFAULT_BUDGETS = {
    # detect_claude.py l.1315-1329
    "ig_samples": 20,
    "ig_steps": 30,
    "ig_baselines": 2,
    "pc_alpha": 0.1,
    "pc_test": "fisherz",
    "cs_samples": 200,
    "cs_orderings": 100,
    "cs_background": 150,
    "epochs": 300,
}


def run_single_seed(config: dict, seed: int, budgets: Optional[dict] = None, verbose: bool = False):
    import torch
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from ..methods.causal_shap_oracle_dag import dag_ordered_shapley_products
    from ..methods.ig_second_difference import compute_ig_second_difference_matrix
    from ..methods.pc_dag_mask import apply_dag_mask, pc_directed_graph
    from ..model import predict_neural_net, train_neural_net

    b = dict(DEFAULT_BUDGETS)
    if budgets:
        b.update(budgets)

    np.random.seed(seed)
    torch.manual_seed(seed)

    df, y, GT, dag_edges = generate_asymmetric_dataset(config, seed=seed)
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    nn_model, scaler_X, scaler_y = train_neural_net(X_train, y_train, n_epochs=b["epochs"], seed=seed)

    def predict_fn(X_batch: np.ndarray):
        return predict_neural_net(nn_model, scaler_X, scaler_y, X_batch)

    test_r2 = float(r2_score(y_test, predict_fn(X_test)))  # added: model quality

    t0 = time.time()
    IG2 = compute_ig_second_difference_matrix(
        nn_model, scaler_X, X_test, X_train,
        n_samples=b["ig_samples"], n_steps=b["ig_steps"], n_baselines=b["ig_baselines"], seed=seed,
    )
    t_ig = time.time() - t0

    t0 = time.time()
    PC_mat = pc_directed_graph(X_train, alpha=b["pc_alpha"], ci_test=b["pc_test"])
    t_pc = time.time() - t0

    t0 = time.time()
    CS = dag_ordered_shapley_products(
        predict_fn, X_train, dag_edges, FEATURE_NAMES,  # NOTE: true DAG = oracle information
        n_samples=b["cs_samples"], max_orderings=b["cs_orderings"], background_size=b["cs_background"], seed=seed,
    )
    t_cs = time.time() - t0

    matrices = {
        "GT": GT,
        "IG2": IG2,
        "IG2xPC": apply_dag_mask(IG2, PC_mat),
        "DAGShapOracle": CS,
        "PC_mask": PC_mat,
    }
    runtimes = {
        "IG2": t_ig,
        "IG2xPC": t_ig + t_pc,  # bug fix: original reported t_pc only
        "DAGShapOracle": t_cs,
        "PC_only": t_pc,
    }

    metrics = {}
    roc_data = {}
    mask = ~np.eye(GT.shape[0], dtype=bool)
    for m in METHODS:
        metrics[m] = all_metrics(matrices[m], GT)
        Mn = normalise(matrices[m])
        GTn = normalise(GT)
        roc_data[m] = ((GTn[mask] > 0).astype(int), Mn[mask])

    extra = {"model_test_r2": test_r2, "pc_n_directed_edges": int(PC_mat.sum())}
    return matrices, runtimes, roc_data, metrics, extra


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")


def run_multi_seed_evaluation(config: dict, seeds: list, budgets: Optional[dict] = None, verbose: bool = True):
    per_seed = []
    pooled_roc = {m: [[], []] for m in METHODS}
    first = None
    for k, seed in enumerate(seeds):
        if verbose:
            print(f"\n=== Seed {seed} (run {k + 1}/{len(seeds)}) ===")
        matrices, runtimes, roc_data, metrics, extra = run_single_seed(config, seed, budgets, verbose)
        if first is None:
            first = matrices
        per_seed.append({"seed": seed, "matrices": matrices, "runtimes": runtimes, "metrics": metrics, "extra": extra})
        for m in METHODS:
            y_true, y_scores = roc_data[m]
            pooled_roc[m][0].extend(y_true.tolist())
            pooled_roc[m][1].extend(y_scores.tolist())

    metric_means = {m: {} for m in METHODS}
    metric_stds = {m: {} for m in METHODS}
    for m in METHODS:
        for key in METRIC_KEYS:
            vals = [r["metrics"][m][key] for r in per_seed]
            metric_means[m][key] = float(np.mean(vals))
            metric_stds[m][key] = _std(vals)
    runtime_means = [float(np.mean([r["runtimes"][m] for r in per_seed])) for m in METHODS]
    runtime_stds = [_std([r["runtimes"][m] for r in per_seed]) for m in METHODS]
    return metric_means, metric_stds, runtime_means, runtime_stds, pooled_roc, per_seed, first


def write_numeric_results(outdir: str, per_seed: list, metric_means: dict, metric_stds: dict, config: dict, budgets: dict, seeds: list):
    import csv

    with open(os.path.join(outdir, "metrics_per_seed.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "method"] + METRIC_KEYS + ["runtime_s", "model_test_r2", "pc_n_directed_edges"])
        for r in per_seed:
            for m in METHODS:
                w.writerow([r["seed"], m] + [r["metrics"][m][k] for k in METRIC_KEYS]
                           + [r["runtimes"][m], r["extra"]["model_test_r2"], r["extra"]["pc_n_directed_edges"]])
    with open(os.path.join(outdir, "metrics_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "metric", "mean", "std_ddof1", "n_seeds"])
        for m in METHODS:
            for k in METRIC_KEYS:
                w.writerow([m, k, metric_means[m][k], metric_stds[m][k], len(per_seed)])
    with open(os.path.join(outdir, "runtimes_per_seed.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "IG2_s", "PC_only_s", "IG2xPC_s", "DAGShapOracle_s"])
        for r in per_seed:
            rt = r["runtimes"]
            w.writerow([r["seed"], rt["IG2"], rt["PC_only"], rt["IG2xPC"], rt["DAGShapOracle"]])
    for r in per_seed:
        np.savez(os.path.join(outdir, f"matrices_seed{r['seed']}.npz"), **r["matrices"])
    with open(os.path.join(outdir, "run_config.json"), "w") as fh:
        json.dump({"seeds": seeds, "scm_config": config, "budgets": budgets}, fh, indent=2)


def run_comparative_analysis(outdir: Optional[str] = None, seeds: Optional[list] = None,
                             base_config: Optional[dict] = None, budgets: Optional[dict] = None):
    from .. import plots

    if outdir is None:
        outdir = "outputs_compare"
    os.makedirs(outdir, exist_ok=True)
    if base_config is None:
        base_config = DEFAULT_CONFIG.copy()
    if seeds is None:
        seeds = [42]
    b = dict(DEFAULT_BUDGETS)
    if budgets:
        b.update(budgets)

    print("=== Multi-seed evaluation ===")
    metric_means, metric_stds, runtime_means, runtime_stds, pooled_roc, per_seed, rep = run_multi_seed_evaluation(
        base_config, seeds, b, verbose=True
    )
    write_numeric_results(outdir, per_seed, metric_means, metric_stds, base_config, b, seeds)

    print("\nIG2 matrix (first seed):\n", rep["IG2"])
    print("\nIG2 x PC matrix (first seed):\n", rep["IG2xPC"])

    n = len(seeds)
    budget_note = (f"IG2: {b['ig_samples']} explicands x {b['ig_baselines']} baselines x {b['ig_steps']} steps; "
                   f"PC: {b['pc_test']}, alpha={b['pc_alpha']}; DAG-Shapley: {b['cs_samples']} rows x <= {b['cs_orderings']} "
                   f"orderings, background {b['cs_background']}")
    print("\n=== Generating figures ===")
    heat = {k: rep[k] for k in ["GT"] + METHODS}
    titles = [plots._one_line(plots.METHOD_LABELS[k]) for k in heat]
    plots.plot_heatmaps(heat, titles, f"{outdir}/fig1_interaction_heatmaps.png")
    plots.plot_asymmetry_comparison(metric_means, metric_stds, f"{outdir}/fig2_asymmetry_comparison.png", n_seeds=n)
    plots.plot_edge_recovery_bar(metric_means, metric_stds, f"{outdir}/fig3_edge_recovery.png", n_seeds=n)
    plots.plot_roc_pr_curves(pooled_roc, f"{outdir}/fig4_roc_curves.png", f"{outdir}/fig5_pr_curves.png")
    plots.plot_directed_graphs(heat, f"{outdir}/fig6_directed_graphs.png")
    plots.plot_diverging_heatmaps(heat, f"{outdir}/fig7_diverging_heatmaps.png")
    plots.plot_rank_consistency(heat, f"{outdir}/fig8_rank_consistency.png")
    plots.plot_radar_chart(metric_means, f"{outdir}/fig9_radar_chart.png", n_seeds=n)
    plots.plot_runtime_complexity(runtime_means, runtime_stds, f"{outdir}/fig10_runtimes.png", n_seeds=n, budgets=budget_note)
    print("\nAll figures and numeric results saved.\n")
    return metric_means, metric_stds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare IG2, IG2 x PC mask and DAG-ordered Shapley (oracle DAG) on the synthetic SCM.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="One or more seeds (default: 42, as in the original run).")
    p.add_argument("--outdir", default="outputs_compare", help="Relative or absolute output directory.")
    p.add_argument("--n-samples", type=int, default=DEFAULT_CONFIG["n_samples"])
    p.add_argument("--noise-level", type=float, default=DEFAULT_CONFIG["noise_level"])
    p.add_argument("--interaction-type", choices=INTERACTION_TYPES, default=DEFAULT_CONFIG["interaction_type"])
    p.add_argument("--confounding", action="store_true", help="Add latent confounder U of X0 and X1 (off in the original run).")
    p.add_argument("--causal-nonlinear", action="store_true", help="Use X2 = 0.6 tanh(2 X0) + ... (off in the original run).")
    for key, val in DEFAULT_BUDGETS.items():
        p.add_argument("--" + key.replace("_", "-"), type=type(val), default=val)
    return p


def config_from_args(args) -> tuple:
    config = {
        "n_samples": args.n_samples,
        "noise_level": args.noise_level,
        "interaction_type": args.interaction_type,
        "include_confounding": args.confounding,
        "causal_nonlinear": args.causal_nonlinear,
    }
    budgets = {k: getattr(args, k) for k in DEFAULT_BUDGETS}
    return config, budgets


def main(argv=None):
    args = build_parser().parse_args(argv)
    config, budgets = config_from_args(args)
    run_comparative_analysis(outdir=args.outdir, seeds=args.seeds, base_config=config, budgets=budgets)


if __name__ == "__main__":
    main()
