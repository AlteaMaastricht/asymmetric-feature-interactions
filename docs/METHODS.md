# Methods: what each matrix actually computes

All matrices are 5 x 5 with `M[i, j]` read as "influence of feature i on feature j"
(rows = influencer, columns = influenced). Line numbers refer to the original
`detect_claude.py` (thesis code folder, last modified 31 May 2026).

## Synthetic SCM and ground truth (`scm.py`; original l.843-927)

```
X0, X1 ~ N(0,1)                    (optional latent confounder U, off by default)
X2 = 0.6 X0 + 0.5 X1 + 0.3 e       (optional 0.6 tanh(2 X0), off by default)
X3 = 0.7 X2 + 0.3 e
X4 = 0.5 X1 + 0.4 e
y  = X0 + 0.5 X1 + 5 X0 sin X2 + 4 X1 tanh X2 + 2 X2 X3 + 1.5 X1 X4 + noise (5 % of Var y)
GT[0,2]=5, GT[1,2]=4, GT[2,3]=2, GT[1,4]=1.5, all other entries 0
```

Every interaction term in `y` joins two features that are also connected by a
DAG edge, and `GT` puts the coefficient on the DAG direction only. The mixed
second differences of `y` are the same in both orders
(`tests/test_scm.py::test_target_second_differences_are_symmetric`), so the
ground truth's asymmetry describes the feature DAG, not the function.

## Model (`model.py`; l.930-986)

MLP 5-128-128-1, Softplus, standardised inputs/targets, Adam 1e-3, 300 epochs,
batch 128, ReduceLROnPlateau, 75/25 train/test split. Its test accuracy was never
measured in the original; the staged runner now records test R².

## IG2: IG second difference (`methods/ig_second_difference.py`; l.989-1098)

Plotted in the original as "Integrated Hessians".

```
D_ij(x, b) = f(b + d_i e_i + d_j e_j) - f(b + d_i e_i) - f(b + d_j e_j) + f(b),   d = x - b
A[i, j]    = max(0, -D_ij)          (computed via single-coordinate IG line integrals, 30 trapezoid steps)
IG2        = mean of A over 20 test explicands x 2 random training baselines
```

* `D_ij = D_ji`: **symmetric by construction**. The original fig7 shows the
  antisymmetric part at the 1e-5 scale.
* The clamp keeps only negative (sub-additive) second differences.
* This is not Integrated Hessians (Janizek et al., 2021), which integrates the
  Hessian along the straight path and scales by `(x_i - b_i)(x_j - b_j)`.
* It does not use conditional integration paths. The conditional-path code
  exists only in `experimental/conditional_path_pipeline.py`, which no
  reported result used.

## IG2 x PC mask (`methods/pc_dag_mask.py`; l.1196-1205, l.1335)

The PC algorithm (causal-learn, Fisher-z, alpha = 0.1) runs on the training
*features*. It keeps the fully oriented edges `i -> j` as a 0/1 mask, and the
result is `IG2 * mask`. All of the directionality comes from causal discovery
on X; the model contributes only magnitudes. In the default SCM, PC can orient
X0 -> X2 <- X1 and X2 -> X3, but not X1 - X4.

## DAG-ordered Shapley products with the oracle DAG (`methods/causal_shap_oracle_dag.py`; l.1101-1186)

Plotted in the original as "Causal SHAP".

* The orderings are the 7 topological orderings of the **true** DAG.
* For each of 200 training rows and each ordering, features are switched from
  a 150-row background to the row's value one at a time. `marginal[k]` is the
  change in the mean prediction.
* `M[i, k]` is the mean of `|marginal[k]| * |marginal[i]|` over the cases where
  i precedes k.
* For every ancestor pair, the reverse entry is exactly 0, so the direction is
  copied from the oracle DAG (`tests/test_causal_shap_oracle_dag.py`).
* The entry is non-zero for an additive model with no interactions.
* This is neither Causal Shapley values (Heskes et al., 2020) nor the `shap`
  package.

## Metrics (`metrics.py`; l.1189-1298)

The metrics are asymmetry, Frobenius similarity, directional accuracy, Spearman
correlation, top-5, a sign-based Hamming distance, precision/recall/F1 at 0.15,
ROC-AUC and AUPRC. GT's support equals the DAG, so the bare DAG adjacency
matrix scores perfectly on direction and edge recovery
(`tests/test_metrics.py`). Asymmetry is computed on un-normalised matrices and
therefore depends on each method's scale.

## Expressing the defense-slide "Causal IG" variant

Slides 27-28 show an IG2 matrix restricted to the **true** DAG edges
(`detection2.py` l.351-399 computes the same `max(0, IG_first - IG_second)`
only for `(i, j)` in `dag_edges`). With this package, that matrix is
`apply_dag_mask(IG2, scm.dag_adjacency())`. It is not part of the default run.
