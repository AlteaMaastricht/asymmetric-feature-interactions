# Provenance

Source: the thesis code folder (`Code /`, no version control). The canonical
script is `detect_claude.py` (1,734 lines, last modified 31 May 2026 13:05). It
produced `outputs_compare/` at 13:15 the same day. The variants
`detect_claude copy.py` (KCI test, larger budgets, 2 Jun) and
`detect_claude_testrun.py` (adds a diagonal-Hessian "IH" and an "AIG" column,
31 May 17:48) were compared line by line. They are not staged.

## File mapping

| Staged file | Source (detect_claude.py lines) | Changes |
|---|---|---|
| `src/asym_interactions/scm.py` | 843-927 | Noise-free target factored into `structural_target` (same formulas, same RNG order). DAG edges/GT weights as constants. Helpers `ground_truth_matrix`, `dag_adjacency` added. |
| `src/asym_interactions/model.py` | 930-986 | `SmallNet(n_in=X_train.shape[1])` instead of the global default (same value, 5). |
| `src/asym_interactions/methods/ig_second_difference.py` | 989-1098 | **Renamed**: `asymmetric_integrated_gradients` -> `ig_second_difference`, `compute_aig_matrix` -> `compute_ig_second_difference_matrix`. Matrix size read from input. torch import guarded. **Added** `second_difference_reference` (NumPy, for tests). |
| `src/asym_interactions/methods/pc_dag_mask.py` | 1196-1205, 1335 | Encoding conversion factored into `directed_mask_from_causallearn_graph`. causal-learn imported lazily. `apply_dag_mask` added (the original multiplied inline). |
| `src/asym_interactions/methods/causal_shap_oracle_dag.py` | 1101-1186 | **Renamed** `causal_shap_asymmetry` -> `dag_ordered_shapley_products` (alias kept). Matrix size read from input. |
| `src/asym_interactions/metrics.py` | 1189-1298 | `N_FEATURES` -> `M.shape[0]`. `METRIC_KEYS` constant. |
| `src/asym_interactions/plots.py` | 834-841, 1392-1685 | Honest method labels. Seed count in titles; no error bars for one seed. **Fixed** fig6 title (said threshold 0.15, code used 0.10). **Fixed** fig7 subtitle (colour description was reversed). Runtime bar label states IG2 + PC. seaborn imported lazily. |
| `src/asym_interactions/experiments/run_compare.py` | 1301-1389, 1688-1734 | CLI (`--seeds`, `--outdir`, SCM options, budgets; defaults = original values). Relative default output dir. **Bug fix**: IG2 x PC runtime = IG2 + PC time (original l.1338 used PC time only). **Bug fix**: consistent sample std (ddof=1) for metrics and runtimes, NaN for one seed (original: population std for metrics = 0, ddof=1 for runtimes = NaN). **Added**: CSV/NPZ/JSON outputs, test R² of the MLP, PC edge count. First-seed matrices reused for figures instead of re-running that seed. |
| `src/asym_interactions/experimental/conditional_path_pipeline.py` | 15-829 | Verbatim copy plus a status header (unused by any result; known issues listed). |
| `tests/*`, `docs/*`, `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore` | — | New. |
| `results/original_detect_claude_2026-05-31/*.png` | `outputs_compare/fig1, fig2, fig7, fig10` | Byte-identical copies, plus an explanatory README. |

No change alters what any method computes. The two bug fixes change only the
reported runtime of the PC-masked method and the error bars.

## Not staged (and why)

| Item | Reason |
|---|---|
| `google_test.py`, root `fig1_heatmaps.png`, `fig2_directed_graphs.png`, `fig3_frobenius_comparison.png` | "Causal discovery" hard-coded to the true DAG (l.78-86). SmoothHess and Causal SHAP baselines mocked as constant symmetric matrices (l.274-279). Panel titles do not match the dict order (l.282). |
| `detection_final.py`, `outputs_dlhcd/` | Superseded branch ("DLHCD", "SmoothHess via Stein's Lemma"; names do not describe the code). |
| `detection.py`, `detection2.py`, `detection3.py`, `detection_8may.py`, `deep_detection.py`, `notears*.py`, `CAP.py`, `CISu.py`, `new.py` | Earlier iterations that copy the same generator/plots. Several hard-code absolute paths. Some produced figures used in the defense slides (see the validation report). Could be added later under `archive/` if needed. |
| `detect_claude copy.py`, `detect_claude_testrun.py` | Variants of the canonical script (differences listed above). |
| `outputs*/`, `outdir = ./`, root PNGs | Generated, single-seed, mostly unattributable. Only four canonical figures are kept, labelled. |
| `.venv/`, `__pycache__/`, `.DS_Store` | Environment and build artefacts. |
| Defense slides (pptx/pdf), third-party papers and book excerpt | Personal names on slides; copyrighted third-party material. |
