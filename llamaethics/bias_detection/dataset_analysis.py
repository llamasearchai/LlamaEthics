"""
Tools for analyzing dataset bias and attribute distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_dataset_bias(dataset, sensitive_attributes, target_attribute):
    """
    Analyze a dataset for potential biases across sensitive attributes.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to analyze
    sensitive_attributes : list
        List of column names for sensitive attributes
    target_attribute : str
        Column name of the target attribute
        
    Returns
    -------
    dict
        Dictionary containing bias analysis results
    """
    results = {}
    
    # Analyze each sensitive attribute
    for attr in sensitive_attributes:
        attr_results = {}
        
        # Get distribution of sensitive attribute
        attr_dist = dataset[attr].value_counts(normalize=True)
        attr_results['distribution'] = attr_dist.to_dict()
        
        # Check conditional distributions of target given sensitive attribute
        target_given_attr = {}
        for value in dataset[attr].unique():
            subset = dataset[dataset[attr] == value]
            if target_attribute in dataset.columns:
                target_dist = subset[target_attribute].value_counts(normalize=True)
                target_given_attr[value] = target_dist.to_dict()
        
        attr_results['target_given_attr'] = target_given_attr
        
        # Calculate statistical bias metrics
        if target_attribute in dataset.columns:
            # Statistical parity difference
            overall_pos_rate = dataset[target_attribute].mean()
            group_pos_rates = []
            
            for value in dataset[attr].unique():
                subset = dataset[dataset[attr] == value]
                group_rate = subset[target_attribute].mean()
                group_pos_rates.append(group_rate)
                
            max_diff = max(group_pos_rates) - min(group_pos_rates)
            attr_results['statistical_parity_difference'] = max_diff
            
            # Disparate impact
            min_rate = min(group_pos_rates)
            max_rate = max(group_pos_rates)
            
            if max_rate > 0:
                disparate_impact = min_rate / max_rate
            else:
                disparate_impact = 1.0
                
            attr_results['disparate_impact'] = disparate_impact
        
        # Check for correlations with other attributes
        correlations = {}
        for col in dataset.columns:
            if col != attr and dataset[col].dtype in [np.int64, np.float64]:
                # For numerical columns, calculate correlation
                corr = dataset[[attr, col]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
        
        attr_results['correlations'] = correlations
        
        results[attr] = attr_results
    
    return results

def plot_attribute_distribution(dataset, attribute, target_attribute=None, figsize=(12, 8)):
    """
    Plot distribution of an attribute and its relationship with a target variable.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to analyze
    attribute : str
        Column name of the attribute to plot
    target_attribute : str, optional
        Column name of the target attribute
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure with distribution plots
    """
    fig = plt.figure(figsize=figsize)
    
    # Get attribute distribution
    attr_counts = dataset[attribute].value_counts().sort_index()
    
    if target_attribute is None:
        # Single plot for attribute distribution
        ax = fig.add_subplot(111)
        ax.bar(attr_counts.index.astype(str), attr_counts.values, color='skyblue')
        ax.set_title(f'Distribution of {attribute}')
        ax.set_xlabel(attribute)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Add count and percentage labels
        total = attr_counts.sum()
        for i, v in enumerate(attr_counts.values):
            ax.text(i, v + 0.01*total, f"{v} ({v/total:.1%})", ha='center')
    
    else:
        # Set up a grid with two plots
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        
        # Plot 1: Attribute distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(attr_counts.index.astype(str), attr_counts.values, color='skyblue')
        ax1.set_title(f'Distribution of {attribute}')
        ax1.set_xlabel(attribute)
        ax1.set_ylabel('Count')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add count and percentage labels
        total = attr_counts.sum()
        for i, v in enumerate(attr_counts.values):
            ax1.text(i, v + 0.01*total, f"{v} ({v/total:.1%})", ha='center')
        
        # Plot 2: Target rate by attribute value
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate positive rate for each attribute value
        positive_rates = []
        attr_values = []
        
        for value in sorted(dataset[attribute].unique()):
            subset = dataset[dataset[attribute] == value]
            if len(subset) > 0:
                pos_rate = subset[target_attribute].mean()
                positive_rates.append(pos_rate)
                attr_values.append(value)
        
        ax2.bar(np.array(attr_values).astype(str), positive_rates, color='lightgreen')
        ax2.set_title(f'{target_attribute} Rate by {attribute}')
        ax2.set_xlabel(attribute)
        ax2.set_ylabel(f'{target_attribute} Rate')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add percentage labels
        for i, v in enumerate(positive_rates):
            ax2.text(i, v + 0.02, f"{v:.1%}", ha='center')
        
        # Plot 3: Target distribution by attribute (heatmap)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Create cross-tabulation of attribute and target
        cross_tab = pd.crosstab(dataset[attribute], dataset[target_attribute], normalize='index')
        
        # Plot heatmap
        sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='.1%', ax=ax3)
        ax3.set_title(f'Distribution of {target_attribute} by {attribute}')
        ax3.set_xlabel(target_attribute)
        ax3.set_ylabel(attribute)
    
    plt.tight_layout()
    return fig

def generate_bias_report(dataset, sensitive_attributes, target_attribute):
    """
    Generate a comprehensive bias report for a dataset.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to analyze
    sensitive_attributes : list
        List of column names for sensitive attributes
    target_attribute : str
        Column name of the target attribute
        
    Returns
    -------
    str
        Text report summarizing potential biases in the dataset
    """
    results = analyze_dataset_bias(dataset, sensitive_attributes, target_attribute)
    
    report = []
    report.append("DATASET BIAS REPORT")
    report.append("=" * 80)
    report.append(f"Dataset shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    report.append(f"Target attribute: {target_attribute}")
    report.append(f"Sensitive attributes: {', '.join(sensitive_attributes)}")
    report.append("")
    
    # Overall target distribution
    target_dist = dataset[target_attribute].value_counts(normalize=True)
    report.append(f"Overall {target_attribute} distribution:")
    for value, prop in target_dist.items():
        report.append(f"  {value}: {prop:.2%}")
    report.append("")
    
    # Analyze each sensitive attribute
    for attr, attr_results in results.items():
        report.append(f"Analysis for {attr}:")
        report.append("-" * 80)
        
        # Distribution of the sensitive attribute
        report.append(f"Distribution of {attr}:")
        for value, prop in attr_results['distribution'].items():
            report.append(f"  {value}: {prop:.2%}")
        report.append("")
        
        # Target distribution given sensitive attribute
        report.append(f"{target_attribute} distribution by {attr}:")
        for attr_value, target_dist in attr_results['target_given_attr'].items():
            report.append(f"  {attr} = {attr_value}:")
            for target_value, prop in target_dist.items():
                report.append(f"    {target_value}: {prop:.2%}")
        report.append("")
        
        # Statistical bias metrics
        if 'statistical_parity_difference' in attr_results:
            spd = attr_results['statistical_parity_difference']
            report.append(f"Statistical parity difference: {spd:.4f}")
            if spd > 0.1:
                report.append("⚠️ POTENTIAL CONCERN: Statistical parity difference > 0.1")
        
        if 'disparate_impact' in attr_results:
            di = attr_results['disparate_impact']
            report.append(f"Disparate impact: {di:.4f}")
            if di < 0.8:
                report.append("⚠️ POTENTIAL CONCERN: Disparate impact < 0.8")
        
        # Correlations with other attributes
        if attr_results['correlations']:
            report.append(f"Top correlations with {attr}:")
            sorted_corrs = sorted(attr_results['correlations'].items(), 
                                 key=lambda x: abs(x[1]), reverse=True)
            for col, corr in sorted_corrs[:5]:  # Show top 5
                report.append(f"  {col}: {corr:.4f}")
        
        report.append("")
    
    return "\n".join(report) 