# asymmetric-feature-interactions

Research code from a BSc thesis on detecting directional ("asymmetric") feature
interactions in a neural network. It compares a gradient-based interaction
matrix, the same matrix masked by a PC-learned causal graph, and a DAG-ordered
Shapley scheme on a synthetic 5-feature structural causal model with a
hand-set ground truth.

> **Read "Methodological limitations" before using any result.** The
> gradient-based measure is symmetric by construction. All of the
> directionality in the reported comparison comes from a causal graph (learned
> by PC, or the true one), and the evaluation rewards recovering that graph.

## 1. Overview

* **Question (thesis).** Can a gradient-based method along conditional paths
  quantify asymmetric feature interactions in deep networks more cheaply than
  causal Shapley approaches?
* **What the code does.** It generates data from a known SCM and trains a
  PyTorch MLP. It then builds three 5 x 5 interaction matrices and scores them
  against a ground-truth matrix with 11 metrics and 10 figures.
* **What the evidence supports.** See [Results](#8-results) and
  [Limitations](#9-limitations). The original runs used one seed and saved no
  numeric results. The comparison design cannot separate "the method detects
  directional interactions" from "the method was given (or learned) the
  feature DAG".

## 2. Problem

Pairwise interaction scores such as Integrated Hessians or Shapley interaction
indices are symmetric: they give one number per feature pair. The thesis asks
whether a directional score can be obtained, i.e. "how much does A change the
importance of B" as distinct from "how much does B change the importance of A".
It also asks whether gradients make this cheaper than coalition sampling.

## 3. Approach

| Key | Name in this repo | Name in the original code/slides | What it computes |
|---|---|---|---|
| `IG2` | IG second difference | "Integrated Hessians (raw)" | Mean over explicands/baselines of `max(0, -D_ij)`, with `D_ij` the mixed finite second difference of the model, obtained from single-coordinate IG line integrals. **Symmetric** (`D_ij = D_ji`). |
| `IG2xPC` | IG2 x PC DAG mask | "Integrated Hessians + PC" | `IG2` multiplied element-wise by the 0/1 matrix of edges that the PC algorithm orients on the feature data. |
| `DAGShapOracle` | DAG-ordered Shapley products, oracle DAG | "Causal SHAP" | For orderings consistent with the **true** DAG: mean of `abs(marginal_k) * abs(marginal_i)` for i before k. Reverse entries of DAG ancestor pairs are always 0. |

Equations and caveats are in [`docs/METHODS.md`](docs/METHODS.md).

## 4. Architecture

```
scm.generate_asymmetric_dataset ──> X, y, GT (= DAG edges), dag_edges
        │
        ├─ model.train_neural_net (MLP 5-128-128-1, Softplus)
        │       │
        │       ├─ methods.ig_second_difference ──────────────> IG2
        │       └─ methods.causal_shap_oracle_dag(dag_edges) ─> DAGShapOracle   (uses TRUE DAG)
        │
        └─ methods.pc_dag_mask (PC on X_train only) ──> mask ──> IG2xPC = IG2 * mask
                                                                │
                                metrics.all_metrics(M, GT) <────┘ ──> plots.* / CSV
```

```
src/asym_interactions/
  scm.py                       synthetic SCM, ground truth, DAG helpers
  model.py                     PyTorch MLP (needs torch)
  methods/
    ig_second_difference.py    IG2 (+ NumPy reference used by tests)
    pc_dag_mask.py             PC (causal-learn) -> directed-edge mask
    causal_shap_oracle_dag.py  DAG-ordered Shapley products
  metrics.py                   11 evaluation metrics
  plots.py                     10 figures
  experiments/run_compare.py   CLI experiment runner
  experimental/conditional_path_pipeline.py   unused conditional-path attribution code (see below)
tests/                         unittest suite (runs without torch; torch tests skip)
docs/METHODS.md, docs/PROVENANCE.md
results/original_detect_claude_2026-05-31/   4 original figures, not regenerated
```

`experimental/conditional_path_pipeline.py` contains the thesis idea of
**conditional** integration paths: Nadaraya–Watson conditional means along a
PC/FCI DAG order, with per-feature attributions weighted by causal descendants.
It is present in the original script but was **never called** by the
comparison that produced the figures. It returns per-feature attributions, not
a pairwise matrix, and has known issues (listed in its header).

## 5. Technologies

Python 3.10+ · PyTorch (autograd) · causal-learn (PC) · NumPy · pandas · SciPy ·
scikit-learn · networkx · matplotlib · seaborn. The original environment used
Python 3.12 with the versions pinned in `requirements.txt`.

## 6. Installation & running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, causal-learn, seaborn, ...
pip install -e .                          # or: export PYTHONPATH=src
```

Tests (no torch needed; torch/causal-learn/seaborn tests skip if missing):

```bash
python -m unittest discover -s tests -v
```

Experiment (needs torch, causal-learn and seaborn):

```bash
# Original configuration (one seed, as in the thesis run)
python -m asym_interactions.experiments.run_compare --seeds 42 --outdir outputs_compare

# What the claims would need: several seeds and stress settings
python -m asym_interactions.experiments.run_compare --seeds 0 1 2 3 4 5 6 7 8 9 --outdir outputs_10seeds
python -m asym_interactions.experiments.run_compare --seeds 0 1 2 3 4 --confounding --causal-nonlinear --outdir outputs_confounded
```

All options: `--help`. These include `--n-samples`, `--noise-level`,
`--interaction-type {nonlinear,threshold,polynomial}`, `--pc-test`,
`--pc-alpha`, `--ig-samples`, `--ig-steps`, `--ig-baselines`, `--cs-samples`,
`--cs-orderings`, `--cs-background` and `--epochs`.

> The full pipeline has **not been executed** in this repository's current form
> because PyTorch was not available during preparation. Run the torch-dependent
> tests first.

## 7. Example usage

Without torch, you can check the structural properties discussed below:

```python
import numpy as np
from asym_interactions import scm, metrics
from asym_interactions.methods.causal_shap_oracle_dag import dag_ordered_shapley_products

df, y, GT, dag = scm.generate_asymmetric_dataset(seed=42)

# The bare DAG adjacency (no model at all) scores perfectly on the direction/edge metrics:
print(metrics.all_metrics(scm.dag_adjacency(), GT))

# The oracle-DAG Shapley scheme is non-zero even for a model with no interactions:
additive = lambda X: X @ np.array([1.0, 2.0, -1.5, 0.5, 1.0])
print(dag_ordered_shapley_products(additive, df.values, dag, n_samples=30, background_size=50).round(2))
```

Each run writes `metrics_per_seed.csv`, `metrics_summary.csv`,
`runtimes_per_seed.csv`, `matrices_seed<k>.npz`, `run_config.json` and
`fig1`–`fig10` to `--outdir`.

## 8. Results

No numeric result files exist from the original work, and the pipeline has not
been re-run here. The only results are four figures from the original
single-seed run of `detect_claude.py` (31 May 2026), copied unchanged into
[`results/original_detect_claude_2026-05-31/`](results/original_detect_claude_2026-05-31/README.md).
They are **original single-seed output of detect_claude.py, not regenerated**,
and they use the original (misleading) method names. The readings below are
taken from those figures:

| Observation | Source |
|---|---|
| The IG2 matrix is symmetric. Its antisymmetric part is on the order of 1e-5 (colour-bar scale). | `fig7_diverging_heatmaps.png`, panel 2 |
| The largest normalised IG2 entry is the X0–X4 pair (1.00), which has no interaction term in y. The true X0–X2 pair scores 0.07. | `fig1_asymmetry_heatmaps.png`, panel 2 |
| PC oriented X0→X2, X1→X2 and X2→X3 but not X1→X4, so IG2 x PC is non-zero only on those three edges. | `fig1_asymmetry_heatmaps.png`, panel 3 |
| Reported runtimes: 1.1 s (IG2), 19.9 s ("IH + PC", which is **PC time only**) and 20.1 s (oracle-DAG Shapley), with unequal sample budgets. | `fig10_runtimes.png`; original l.1338 |

Tests in `tests/` demonstrate the structural properties the README relies on:
symmetry of the second difference, oracle-copied direction, and that the bare
DAG adjacency scores perfectly on the evaluation. They use synthetic inputs and
are not experimental results.

## 9. Limitations

### Methodological limitations

1. **The gradient measure is symmetric by construction.** IG2 equals
   `max(0, -D_ij)` with `D_ij = D_ji`, so it cannot express direction. The
   clamp also discards all positive (super-additive) interaction. It is not
   Integrated Hessians, and it does not use conditional integration paths.
2. **Direction comes only from a causal graph.** In `IG2xPC`, every asymmetry
   comes from PC run on the features, which never sees the model. The
   explanation contributes magnitudes only.
3. **The baseline gets the oracle DAG.** `DAGShapOracle` is given the true DAG
   of the SCM. Its direction is copied from that DAG (reverse entries of
   ancestor pairs are exactly 0), and its magnitude is a product of marginal
   contributions that is non-zero without any interaction. The comparison
   therefore pits a learned graph against the true graph rather than one
   interaction method against another.
4. **The evaluation rewards DAG recovery.** The ground truth is non-zero exactly
   on the DAG edges and in the DAG direction, while the target's interaction
   terms are symmetric functions. The binary DAG adjacency, with no model
   information, gets directional accuracy 1, ROC-AUC 1 and AUPRC 1
   (`tests/test_metrics.py`).
5. **A single seed.** The original run used `seeds = [42]`. The
   "mean ± std over seeds" bars were zero (metrics) or NaN (runtimes). Some
   earlier figures say "5 seeds" while their scripts ran one.
6. **The runtime comparison is not like-for-like.**
   * IG2 is evaluated on 20 explicands x 2 baselines x 30 steps; the Shapley
     scheme on 200 rows x 7 orderings x 5 coalitions x 150 background rows.
   * The original "IH + PC" bar showed PC time only (fixed in the runner).
   * The defense deck's "1.4 s vs 69.4 s" comes from an earlier script whose IG
     was computed only on the 4 true DAG edges.
7. **Other metric caveats.**
   * Asymmetry is computed on raw matrices, so its scale depends on the method.
   * Top-5 accuracy has a maximum of 0.8.
   * A tie counts as a wrong direction.
8. **Scope.** One 5-feature toy SCM, no real data, and a model whose fit
   quality was never measured (the runner now logs test R²).

### What would be needed to support the thesis claims

* A **genuinely directional** measure, e.g. the asymmetric path variant from
  earlier iterations (one feature held fixed while the other is integrated)
  or the conditional-path code in `experimental/`, tested to produce
  asymmetry on a model **without** any DAG mask.
* A ground truth whose **direction is a property of the function** (e.g.
  `y = g(X_i) · h(X_j)` designs where the directional quantity is defined
  analytically), not of the feature DAG. Also controls where the DAG and the
  interaction direction disagree.
* The **same graph for every method**: either all get the PC graph or all get
  the oracle graph, plus a "DAG adjacency only" baseline.
* **Equal sample budgets** (same explicands, baselines and background) and
  timings that include every stage.
* **≥ 10 seeds** with confidence intervals, sweeps over noise, confounding and
  interaction type, and the numeric CSVs committed alongside the figures.
* Report the explained model's held-out accuracy.

## 10. Reproducibility

* **Data.** Synthetic only, generated in `scm.py` from a seed (NumPy global
  RNG). There is nothing to download.
* **Seeds.** `--seeds` controls data, the train/test split, network
  initialisation, baselines and orderings. The original figures used seed 42.
* **What can be regenerated.** Everything in `run_compare` once torch,
  causal-learn and seaborn are installed. This has **not** been verified here.
  CPU float results may still differ slightly across library versions.
* **What cannot be regenerated.** The defense-slide figures came from other
  scripts or from versions of the code that are no longer on disk (see
  `docs/PROVENANCE.md`). The original `outputs_compare/fig10` timing suggests
  the run used a different CI test than the saved file.
* **Verified here.** `python -m unittest discover -s tests` passes 32 tests,
  with 4 skipped (torch / causal-learn / seaborn not installed).

## 11. Context, attribution & licence

* **Context.** Individual BSc thesis (Business Analytics), Maastricht
  University, 2026. Single author; the supervisor is not named here.
  This repository holds the code only. The written thesis and the defense
  slides are not included.
* **Origin of the code.** It was consolidated from the thesis script
  `detect_claude.py` without changing what the methods compute. Renames, bug
  fixes and additions are listed in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).
  Earlier iterations (≈15 scripts) and a script with mocked baselines were
  deliberately left out.
* **AI assistance.** Parts of this code were written with AI coding assistants (ChatGPT/Claude); all design decisions and results were reviewed by the author.
  The original file names (`detect_claude*.py`) indicate substantial
  AI-generated code.
* **Methods referenced.** Integrated Gradients (Sundararajan et al., 2017),
  Integrated Hessians (Janizek et al., 2021), Causal Shapley values
  (Heskes et al., 2020), the PC algorithm via causal-learn. No third-party code
  or data is included.
* License: MIT — see [`LICENSE`](LICENSE).
