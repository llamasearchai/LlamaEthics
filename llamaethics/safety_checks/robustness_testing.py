"""
Robustness testing module for evaluating model stability and reliability.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def test_robustness(model, data, target=None, noise_types=['gaussian', 'uniform'], noise_levels=[0.01, 0.05, 0.1, 0.2], n_samples=1000, random_state=42):
    """
    Test model robustness by adding different types of noise to input data.
    
    Parameters
    ----------
    model : object
        The trained model to test
    data : array-like
        Input data for testing
    target : array-like, optional
        Target values for calculating performance metrics
    noise_types : list, optional
        Types of noise to add ('gaussian', 'uniform', 'salt_pepper')
    noise_levels : list, optional
        Levels of noise to add (as proportion of data range)
    n_samples : int, optional
        Number of samples to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing robustness test results
    """
    np.random.seed(random_state)
    
    # Convert to numpy arrays
    X = data.values if isinstance(data, pd.DataFrame) else data
    
    # Subsample if needed
    if n_samples < X.shape[0]:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
        if target is not None:
            y = target[indices]
    else:
        if target is not None:
            y = target
    
    # Get baseline predictions
    baseline_preds = model.predict(X)
    
    # Calculate baseline metrics if target is provided
    baseline_metrics = {}
    if target is not None:
        baseline_metrics['accuracy'] = accuracy_score(y, baseline_preds)
        if len(np.unique(y)) == 2:  # Binary classification
            baseline_metrics['precision'] = precision_score(y, baseline_preds, average='binary')
            baseline_metrics['recall'] = recall_score(y, baseline_preds, average='binary')
            baseline_metrics['f1'] = f1_score(y, baseline_preds, average='binary')
        else:  # Multi-class classification
            baseline_metrics['precision'] = precision_score(y, baseline_preds, average='weighted')
            baseline_metrics['recall'] = recall_score(y, baseline_preds, average='weighted')
            baseline_metrics['f1'] = f1_score(y, baseline_preds, average='weighted')
    
    # Calculate data range for scaling noise
    data_range = np.max(X) - np.min(X)
    
    # Test different noise types and levels
    results = {
        'baseline': {
            'predictions': baseline_preds,
            'metrics': baseline_metrics
        },
        'noise_tests': {}
    }
    
    for noise_type in noise_types:
        noise_results = {}
        
        for level in noise_levels:
            # Generate noisy data
            noise_amount = level * data_range
            
            if noise_type == 'gaussian':
                noisy_X = X + np.random.normal(0, noise_amount, X.shape)
            elif noise_type == 'uniform':
                noisy_X = X + np.random.uniform(-noise_amount, noise_amount, X.shape)
            elif noise_type == 'salt_pepper':
                # Salt and pepper noise
                mask = np.random.random(X.shape) < level
                noisy_X = X.copy()
                noisy_X[mask] = np.random.choice([np.min(X), np.max(X)], mask.sum())
            
            # Get predictions on noisy data
            noisy_preds = model.predict(noisy_X)
            
            # Calculate prediction consistency
            consistency = np.mean(noisy_preds == baseline_preds)
            
            # Calculate metrics if target is provided
            metrics = {}
            if target is not None:
                metrics['accuracy'] = accuracy_score(y, noisy_preds)
                
                if len(np.unique(y)) == 2:  # Binary classification
                    metrics['precision'] = precision_score(y, noisy_preds, average='binary')
                    metrics['recall'] = recall_score(y, noisy_preds, average='binary')
                    metrics['f1'] = f1_score(y, noisy_preds, average='binary')
                else:  # Multi-class classification
                    metrics['precision'] = precision_score(y, noisy_preds, average='weighted')
                    metrics['recall'] = recall_score(y, noisy_preds, average='weighted')
                    metrics['f1'] = f1_score(y, noisy_preds, average='weighted')
                
                # Calculate metric changes
                metric_changes = {}
                for metric, value in metrics.items():
                    change = value - baseline_metrics[metric]
                    metric_changes[metric] = change
                    metric_changes[f'{metric}_percent'] = (change / baseline_metrics[metric]) * 100
            
            # Store results
            noise_results[level] = {
                'consistency': consistency,
                'metrics': metrics,
                'metric_changes': metric_changes if target is not None else {}
            }
        
        results['noise_tests'][noise_type] = noise_results
    
    return results

def test_edge_cases(model, data, target=None, extremes_percentile=5, outliers_factor=3, n_clusters=5, random_state=42):
    """
    Test model performance on edge cases like extremes, outliers, and rare combinations.
    
    Parameters
    ----------
    model : object
        The trained model to test
    data : array-like
        Input data for testing
    target : array-like, optional
        Target values for calculating performance metrics
    extremes_percentile : int, optional
        Percentile to use for defining extreme values
    outliers_factor : float, optional
        Factor of standard deviations to use for defining outliers
    n_clusters : int, optional
        Number of clusters to use for finding rare combinations
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing edge case test results
    """
    from sklearn.cluster import KMeans
    
    np.random.seed(random_state)
    
    # Convert to pandas DataFrame for easier manipulation
    if isinstance(data, pd.DataFrame):
        X = data.copy()
    else:
        X = pd.DataFrame(data)
    
    # Store original data
    X_orig = X.copy()
    
    # Store baseline predictions
    baseline_preds = model.predict(X)
    
    results = {
        'baseline': {
            'predictions': baseline_preds,
            'metrics': {}
        },
        'edge_cases': {}
    }
    
    # Calculate baseline metrics if target is provided
    if target is not None:
        results['baseline']['metrics']['accuracy'] = accuracy_score(target, baseline_preds)
        
        if len(np.unique(target)) == 2:  # Binary classification
            results['baseline']['metrics']['precision'] = precision_score(target, baseline_preds, average='binary')
            results['baseline']['metrics']['recall'] = recall_score(target, baseline_preds, average='binary')
            results['baseline']['metrics']['f1'] = f1_score(target, baseline_preds, average='binary')
        else:  # Multi-class classification
            results['baseline']['metrics']['precision'] = precision_score(target, baseline_preds, average='weighted')
            results['baseline']['metrics']['recall'] = recall_score(target, baseline_preds, average='weighted')
            results['baseline']['metrics']['f1'] = f1_score(target, baseline_preds, average='weighted')
    
    # 1. Test extreme values
    extreme_results = {
        'low': {},
        'high': {}
    }
    
    # Find extremes for each feature
    for col in X.columns:
        # Low extremes
        low_threshold = np.percentile(X[col], extremes_percentile)
        low_extremes = X_orig[X_orig[col] <= low_threshold]
        
        if len(low_extremes) > 0:
            low_preds = model.predict(low_extremes)
            
            extreme_results['low'][col] = {
                'count': len(low_extremes),
                'threshold': low_threshold
            }
            
            if target is not None:
                low_indices = X_orig[X_orig[col] <= low_threshold].index
                low_targets = target[low_indices]
                
                extreme_results['low'][col]['accuracy'] = accuracy_score(low_targets, low_preds)
        
        # High extremes
        high_threshold = np.percentile(X[col], 100 - extremes_percentile)
        high_extremes = X_orig[X_orig[col] >= high_threshold]
        
        if len(high_extremes) > 0:
            high_preds = model.predict(high_extremes)
            
            extreme_results['high'][col] = {
                'count': len(high_extremes),
                'threshold': high_threshold
            }
            
            if target is not None:
                high_indices = X_orig[X_orig[col] >= high_threshold].index
                high_targets = target[high_indices]
                
                extreme_results['high'][col]['accuracy'] = accuracy_score(high_targets, high_preds)
    
    results['edge_cases']['extremes'] = extreme_results
    
    # 2. Test outliers
    outlier_results = {}
    
    # Find outliers for each feature
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        
        # Outliers are values more than outliers_factor standard deviations from the mean
        outliers = X_orig[(X_orig[col] < mean - outliers_factor * std) | 
                          (X_orig[col] > mean + outliers_factor * std)]
        
        if len(outliers) > 0:
            outlier_preds = model.predict(outliers)
            
            outlier_results[col] = {
                'count': len(outliers),
                'lower_threshold': mean - outliers_factor * std,
                'upper_threshold': mean + outliers_factor * std
            }
            
            if target is not None:
                outlier_indices = outliers.index
                outlier_targets = target[outlier_indices]
                
                outlier_results[col]['accuracy'] = accuracy_score(outlier_targets, outlier_preds)
    
    results['edge_cases']['outliers'] = outlier_results
    
    # 3. Test rare combinations using clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X)
    
    # Find the smallest cluster (rare combinations)
    cluster_counts = np.bincount(clusters)
    rarest_cluster = np.argmin(cluster_counts)
    rare_indices = np.where(clusters == rarest_cluster)[0]
    
    if len(rare_indices) > 0:
        rare_samples = X_orig.iloc[rare_indices]
        rare_preds = model.predict(rare_samples)
        
        rare_results = {
            'count': len(rare_samples),
            'proportion': len(rare_samples) / len(X)
        }
        
        if target is not None:
            rare_targets = target[rare_indices]
            rare_results['accuracy'] = accuracy_score(rare_targets, rare_preds)
        
        results['edge_cases']['rare_combinations'] = rare_results
    
    return results

def plot_robustness_results(results, metric='accuracy', figsize=(12, 8)):
    """
    Plot robustness test results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing robustness test results from test_robustness
    metric : str, optional
        Metric to plot ('accuracy', 'precision', 'recall', 'f1', or 'consistency')
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    Figure
        Matplotlib figure with robustness plots
    """
    plt.figure(figsize=figsize)
    
    noise_types = list(results['noise_tests'].keys())
    noise_levels = list(results['noise_tests'][noise_types[0]].keys())
    
    # Set up colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Plot baseline
    if metric != 'consistency' and metric in results['baseline']['metrics']:
        baseline_value = results['baseline']['metrics'][metric]
        plt.axhline(y=baseline_value, color='black', linestyle='--', label='Baseline')
    
    # Plot metrics for each noise type
    for i, noise_type in enumerate(noise_types):
        metric_values = []
        
        for level in noise_levels:
            if metric == 'consistency':
                value = results['noise_tests'][noise_type][level]['consistency']
            else:
                value = results['noise_tests'][noise_type][level]['metrics'][metric]
            
            metric_values.append(value)
        
        plt.plot(noise_levels, metric_values, marker='o', color=colors[i % len(colors)], 
                label=f'{noise_type.capitalize()} Noise')
    
    plt.xlabel('Noise Level')
    plt.ylabel(metric.capitalize())
    plt.title(f'Effect of Noise on {metric.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def generate_robustness_report(results, format='text'):
    """
    Generate a comprehensive report from robustness test results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing robustness test results from test_robustness
    format : str, optional
        Report format ('text', 'html', or 'dict')
        
    Returns
    -------
    str or dict
        The robustness report
    """
    if format == 'dict':
        return results
    
    noise_types = list(results['noise_tests'].keys())
    noise_levels = list(results['noise_tests'][noise_types[0]].keys())
    
    # Create report
    if format == 'text':
        report = []
        report.append("MODEL ROBUSTNESS REPORT")
        report.append("=" * 80)
        
        # Baseline metrics
        if results['baseline']['metrics']:
            report.append("\nBASELINE METRICS")
            report.append("-" * 80)
            for metric, value in results['baseline']['metrics'].items():
                report.append(f"{metric.capitalize()}: {value:.4f}")
        
        # Noise test results
        report.append("\nNOISE TEST RESULTS")
        report.append("-" * 80)
        
        for noise_type in noise_types:
            report.append(f"\n{noise_type.upper()} NOISE")
            
            # Table header
            header = ["Noise Level", "Consistency"]
            if results['baseline']['metrics']:
                for metric in results['baseline']['metrics'].keys():
                    header.append(f"{metric.capitalize()}")
                    header.append(f"{metric.capitalize()} Change")
            
            report.append("\t".join(header))
            report.append("-" * 100)
            
            # Table rows
            for level in noise_levels:
                level_result = results['noise_tests'][noise_type][level]
                
                row = [f"{level:.2f}", f"{level_result['consistency']:.4f}"]
                
                if level_result['metrics']:
                    for metric in results['baseline']['metrics'].keys():
                        value = level_result['metrics'][metric]
                        change = level_result['metric_changes'][metric]
                        
                        row.append(f"{value:.4f}")
                        row.append(f"{change:.4f} ({level_result['metric_changes'][f'{metric}_percent']:.1f}%)")
                
                report.append("\t".join(row))
        
        # Summary
        report.append("\nROBUSTNESS SUMMARY")
        report.append("-" * 80)
        
        # Find highest noise level with acceptable performance (less than 5% drop)
        for noise_type in noise_types:
            max_level = 0
            for level in sorted(noise_levels):
                consistency = results['noise_tests'][noise_type][level]['consistency']
                
                if consistency < 0.95:  # Less than 95% consistency
                    break
                
                max_level = level
            
            report.append(f"{noise_type.capitalize()} Noise: Robust up to {max_level:.2f} noise level (95% consistency)")
        
        return "\n".join(report)
    
    elif format == 'html':
        # Simple HTML report
        html = ["<html><body>"]
        html.append("<h1>Model Robustness Report</h1>")
        
        # Baseline metrics
        if results['baseline']['metrics']:
            html.append("<h2>Baseline Metrics</h2>")
            html.append("<table border='1'><tr><th>Metric</th><th>Value</th></tr>")
            
            for metric, value in results['baseline']['metrics'].items():
                html.append(f"<tr><td>{metric.capitalize()}</td><td>{value:.4f}</td></tr>")
            
            html.append("</table>")
        
        # Noise test results
        html.append("<h2>Noise Test Results</h2>")
        
        for noise_type in noise_types:
            html.append(f"<h3>{noise_type.capitalize()} Noise</h3>")
            
            # Table
            html.append("<table border='1'><tr><th>Noise Level</th><th>Consistency</th>")
            
            if results['baseline']['metrics']:
                for metric in results['baseline']['metrics'].keys():
                    html.append(f"<th>{metric.capitalize()}</th><th>{metric.capitalize()} Change</th>")
            
            html.append("</tr>")
            
            for level in noise_levels:
                level_result = results['noise_tests'][noise_type][level]
                
                html.append(f"<tr><td>{level:.2f}</td><td>{level_result['consistency']:.4f}</td>")
                
                if level_result['metrics']:
                    for metric in results['baseline']['metrics'].keys():
                        value = level_result['metrics'][metric]
                        change = level_result['metric_changes'][metric]
                        percent = level_result['metric_changes'][f'{metric}_percent']
                        
                        html.append(f"<td>{value:.4f}</td><td>{change:.4f} ({percent:.1f}%)</td>")
                
                html.append("</tr>")
            
            html.append("</table>")
        
        html.append("</body></html>")
        
        return "\n".join(html)
    
    return "Unsupported format" 