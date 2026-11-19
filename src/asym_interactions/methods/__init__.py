"""Interaction-matrix methods, each named after what it actually computes.

* :mod:`.ig_second_difference` -- symmetric IG second-difference matrix (needs torch)
* :mod:`.pc_dag_mask` -- PC-learned directed-edge mask (needs causal-learn for PC itself)
* :mod:`.causal_shap_oracle_dag` -- DAG-ordered Shapley marginal products, given a DAG (NumPy + networkx)
"""
