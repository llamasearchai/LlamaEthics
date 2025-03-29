"""
Core metrics for quantifying bias in AI systems.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def detect_bias(dataset, sensitive_attribute, target_attribute, model=None, predictions=None):
    """
    Detect bias in a dataset or model predictions with respect to a sensitive attribute.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to analyze for bias
    sensitive_attribute : str
        Column name of the sensitive attribute (e.g., 'gender', 'race')
    target_attribute : str
        Column name of the target attribute (e.g., 'income', 'hired')
    model : object, optional
        Trained model with a predict method
    predictions : array-like, optional
        Model predictions if already computed
        
    Returns
    -------
    dict
        Dictionary containing fairness metrics
    """
    if model is not None and predictions is None:
        # Get features excluding the target
        X = dataset.drop(columns=[target_attribute])
        predictions = model.predict(X)
    
    # If neither model nor predictions provided, use actual values in dataset
    if predictions is None:
        predictions = dataset[target_attribute].values
    
    # Get unique values of sensitive attribute
    sensitive_values = dataset[sensitive_attribute].unique()
    
    # Calculate fairness metrics
    metrics = {}
    metrics['demographic_parity'] = demographic_parity(dataset, sensitive_attribute, predictions)
    metrics['equal_opportunity'] = equal_opportunity(dataset, sensitive_attribute, target_attribute, predictions)
    metrics['disparate_impact'] = disparate_impact(dataset, sensitive_attribute, predictions)
    
    # Calculate subgroup metrics
    metrics['subgroup_metrics'] = {}
    for value in sensitive_values:
        subset = dataset[dataset[sensitive_attribute] == value]
        subset_indices = dataset.index[dataset[sensitive_attribute] == value].tolist()
        subset_preds = [predictions[i] for i in subset_indices] if isinstance(predictions, list) else predictions[subset_indices]
        subset_actual = subset[target_attribute].values
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(subset_actual, subset_preds, labels=[0, 1]).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['subgroup_metrics'][value] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': len(subset)
        }
    
    return metrics

def demographic_parity(dataset, sensitive_attribute, predictions):
    """
    Calculate demographic parity - the difference in positive prediction rates between groups.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to analyze
    sensitive_attribute : str
        Column name of the sensitive attribute
    predictions : array-like
        Model predictions
        
    Returns
    -------
    dict
        Dictionary with demographic parity metrics
    """
    sensitive_values = dataset[sensitive_attribute].unique()
    positive_rates = {}
    
    for value in sensitive_values:
        indices = dataset.index[dataset[sensitive_attribute] == value].tolist()
        subset_preds = [predictions[i] for i in indices] if isinstance(predictions, list) else predictions[indices]
        positive_rate = np.mean(subset_preds)
        positive_rates[value] = positive_rate
    
    # Calculate max difference between any two groups
    values = list(positive_rates.values())
    max_diff = max(values) - min(values)
    
    return {
        'positive_rates': positive_rates,
        'max_difference': max_diff
    }

def equal_opportunity(dataset, sensitive_attribute, target_attribute, predictions):
    """
    Calculate equal opportunity - the difference in true positive rates between groups.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to analyze
    sensitive_attribute : str
        Column name of the sensitive attribute
    target_attribute : str
        Column name of the target attribute
    predictions : array-like
        Model predictions
        
    Returns
    -------
    dict
        Dictionary with equal opportunity metrics
    """
    sensitive_values = dataset[sensitive_attribute].unique()
    true_positive_rates = {}
    
    for value in sensitive_values:
        subset = dataset[dataset[sensitive_attribute] == value]
        subset_indices = dataset.index[dataset[sensitive_attribute] == value].tolist()
        subset_preds = [predictions[i] for i in subset_indices] if isinstance(predictions, list) else predictions[subset_indices]
        subset_actual = subset[target_attribute].values
        
        # Calculate true positive rate for this group
        positives = (subset_actual == 1)
        if np.sum(positives) > 0:
            true_positives = np.logical_and(subset_preds == 1, subset_actual == 1)
            tpr = np.sum(true_positives) / np.sum(positives)
        else:
            tpr = 0
            
        true_positive_rates[value] = tpr
    
    # Calculate max difference between any two groups
    values = list(true_positive_rates.values())
    max_diff = max(values) - min(values)
    
    return {
        'true_positive_rates': true_positive_rates,
        'max_difference': max_diff
    }

def disparate_impact(dataset, sensitive_attribute, predictions):
    """
    Calculate disparate impact - the ratio of positive prediction rates between groups.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to analyze
    sensitive_attribute : str
        Column name of the sensitive attribute
    predictions : array-like
        Model predictions
        
    Returns
    -------
    dict
        Dictionary with disparate impact metrics
    """
    sensitive_values = dataset[sensitive_attribute].unique()
    positive_rates = {}
    
    for value in sensitive_values:
        indices = dataset.index[dataset[sensitive_attribute] == value].tolist()
        subset_preds = [predictions[i] for i in indices] if isinstance(predictions, list) else predictions[indices]
        positive_rate = np.mean(subset_preds)
        positive_rates[value] = positive_rate
    
    # Calculate ratios
    ratios = {}
    reference_value = list(positive_rates.keys())[0]
    reference_rate = positive_rates[reference_value]
    
    for value, rate in positive_rates.items():
        if rate > 0 and reference_rate > 0:
            ratios[f"{value}/{reference_value}"] = rate / reference_rate
    
    # Calculate min ratio (worst case)
    min_ratio = min(ratios.values()) if ratios else 0
    
    return {
        'positive_rates': positive_rates,
        'impact_ratios': ratios,
        'min_ratio': min_ratio
    }

def fairness_report(metrics, threshold=0.2, output_format='text'):
    """
    Generate a comprehensive fairness report based on metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing fairness metrics from detect_bias
    threshold : float, optional
        Threshold for highlighting concerning disparities
    output_format : str, optional
        Format of the report ('text', 'dict', or 'plot')
        
    Returns
    -------
    str or dict or Figure
        Fairness report in the requested format
    """
    if output_format == 'dict':
        return metrics
    
    if output_format == 'plot':
        return plot_fairness_metrics(metrics)
    
    # Default to text report
    report = []
    report.append("FAIRNESS REPORT")
    report.append("=" * 80)
    
    # Demographic parity
    dp = metrics['demographic_parity']
    report.append("\nDEMOGRAPHIC PARITY")
    report.append("-" * 80)
    for group, rate in dp['positive_rates'].items():
        report.append(f"Group '{group}': {rate:.4f} positive prediction rate")
    report.append(f"Maximum difference: {dp['max_difference']:.4f}")
    if dp['max_difference'] > threshold:
        report.append("⚠️ POTENTIAL CONCERN: Large difference in positive prediction rates")
    
    # Equal opportunity
    eo = metrics['equal_opportunity']
    report.append("\nEQUAL OPPORTUNITY")
    report.append("-" * 80)
    for group, rate in eo['true_positive_rates'].items():
        report.append(f"Group '{group}': {rate:.4f} true positive rate")
    report.append(f"Maximum difference: {eo['max_difference']:.4f}")
    if eo['max_difference'] > threshold:
        report.append("⚠️ POTENTIAL CONCERN: Large difference in true positive rates")
    
    # Disparate impact
    di = metrics['disparate_impact']
    report.append("\nDISPARATE IMPACT")
    report.append("-" * 80)
    for ratio_name, ratio_value in di['impact_ratios'].items():
        report.append(f"Ratio {ratio_name}: {ratio_value:.4f}")
    report.append(f"Minimum ratio: {di['min_ratio']:.4f}")
    if di['min_ratio'] < 0.8:
        report.append("⚠️ POTENTIAL CONCERN: Ratio below 0.8 may indicate disparate impact")
    
    # Subgroup metrics
    report.append("\nSUBGROUP PERFORMANCE")
    report.append("-" * 80)
    subgroup_metrics = metrics['subgroup_metrics']
    for group, group_metrics in subgroup_metrics.items():
        report.append(f"Group '{group}' (n={group_metrics['count']})")
        report.append(f"  Accuracy:  {group_metrics['accuracy']:.4f}")
        report.append(f"  Precision: {group_metrics['precision']:.4f}")
        report.append(f"  Recall:    {group_metrics['recall']:.4f}")
        report.append(f"  F1 Score:  {group_metrics['f1']:.4f}")
    
    # Find largest performance gaps
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics_to_compare:
        values = [m[metric] for m in subgroup_metrics.values()]
        max_diff = max(values) - min(values) if values else 0
        if max_diff > threshold:
            report.append(f"⚠️ POTENTIAL CONCERN: Large difference in {metric} between groups")
    
    return "\n".join(report)

def plot_fairness_metrics(metrics):
    """
    Create visualizations of fairness metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing fairness metrics from detect_bias
        
    Returns
    -------
    Figure
        Matplotlib figure with fairness visualizations
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot positive prediction rates (demographic parity)
    dp = metrics['demographic_parity']
    groups = list(dp['positive_rates'].keys())
    values = list(dp['positive_rates'].values())
    
    axes[0, 0].bar(groups, values, color='skyblue')
    axes[0, 0].set_title('Positive Prediction Rates by Group')
    axes[0, 0].set_ylabel('Positive Rate')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Plot true positive rates (equal opportunity)
    eo = metrics['equal_opportunity']
    groups = list(eo['true_positive_rates'].keys())
    values = list(eo['true_positive_rates'].values())
    
    axes[0, 1].bar(groups, values, color='lightgreen')
    axes[0, 1].set_title('True Positive Rates by Group')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(values):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Plot disparate impact ratios
    di = metrics['disparate_impact']
    if di['impact_ratios']:
        ratio_names = list(di['impact_ratios'].keys())
        ratio_values = list(di['impact_ratios'].values())
        
        axes[1, 0].bar(ratio_names, ratio_values, color='salmon')
        axes[1, 0].set_title('Disparate Impact Ratios')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].axhline(y=0.8, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].text(0, 0.81, '0.8 threshold', color='r')
        for i, v in enumerate(ratio_values):
            axes[1, 0].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Plot subgroup performance comparison
    subgroup_metrics = metrics['subgroup_metrics']
    groups = list(subgroup_metrics.keys())
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metrics_data = {metric: [subgroup_metrics[group][metric] for group in groups] for metric in metrics_to_plot}
    
    bar_width = 0.2
    index = np.arange(len(groups))
    
    for i, metric in enumerate(metrics_to_plot):
        axes[1, 1].bar(index + i*bar_width, metrics_data[metric], 
                       bar_width, label=metric.capitalize())
    
    axes[1, 1].set_title('Performance Metrics by Group')
    axes[1, 1].set_xticks(index + bar_width * 1.5)
    axes[1, 1].set_xticklabels(groups)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig 