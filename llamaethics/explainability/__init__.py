"""
Explainability module for making AI model decisions understandable to humans.
"""

from .model_explanations import explain_model, explain_prediction, plot_feature_importance
from .lime_explanations import explain_instance, explain_text
from .global_explanations import explain_global, feature_importance_summary 