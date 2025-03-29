"""
Bias detection module for identifying and quantifying biases in AI systems.
"""

from .bias_metrics import detect_bias, fairness_report, demographic_parity, equal_opportunity
from .dataset_analysis import analyze_dataset_bias, plot_attribute_distribution 