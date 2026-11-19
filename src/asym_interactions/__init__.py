"""asym_interactions: research code for a BSc thesis on directional ("asymmetric")
feature-interaction attribution in neural networks, evaluated on a synthetic SCM.

Modules that need only NumPy/SciPy/scikit-learn/networkx: ``scm``, ``metrics``,
``methods.causal_shap_oracle_dag``, ``methods.pc_dag_mask`` (helpers; PC itself
needs causal-learn), ``methods.ig_second_difference.second_difference_reference``.
``model`` and the IG2 computation need PyTorch; figures need matplotlib (+ seaborn
for heatmaps).
"""

__version__ = "0.1.0"
