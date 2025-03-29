"""
Core functions for explaining machine learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
import seaborn as sns

def explain_model(model, data, feature_names=None, class_names=None, model_type=None):
    """
    Generate SHAP explanations for a machine learning model.
    
    Parameters
    ----------
    model : object
        The trained model to explain
    data : array-like
        The data to use for explaining the model
    feature_names : list, optional
        Names of the features
    class_names : list, optional
        Names of the target classes
    model_type : str, optional
        Type of model ('tree', 'linear', or 'deep'). If None, will try to infer.
        
    Returns
    -------
    dict
        Dictionary containing SHAP values and explanation objects
    """
    # Convert to DataFrame if it's not already
    if isinstance(data, pd.DataFrame):
        if feature_names is None:
            feature_names = data.columns.tolist()
        data_values = data.values
    else:
        data_values = data
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(data.shape[1])]
            
    # Try to infer model type if not specified
    if model_type is None:
        if hasattr(model, 'feature_importances_') or hasattr(model, 'estimators_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            model_type = 'deep'  # Fallback
    
    # Create appropriate explainer based on model type
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, data_values)
    elif model_type == 'deep':
        explainer = shap.DeepExplainer(model, data_values[:100])  # Use a subset for deep models
    else:
        explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, 
                                       data_values[:100])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(data_values)
    
    # Organize results
    explanation = {
        'shap_values': shap_values,
        'explainer': explainer,
        'feature_names': feature_names,
        'class_names': class_names,
        'model_type': model_type,
        'data': data,
    }
    
    return explanation

def explain_prediction(model, instance, feature_names=None, class_names=None, model_type=None, num_features=10):
    """
    Explain a single prediction from a model.
    
    Parameters
    ----------
    model : object
        The trained model to explain
    instance : array-like
        The single instance to explain
    feature_names : list, optional
        Names of the features
    class_names : list, optional
        Names of the target classes
    model_type : str, optional
        Type of model ('tree', 'linear', or 'deep')
    num_features : int, optional
        Number of top features to show in the explanation
        
    Returns
    -------
    dict
        Dictionary containing the prediction explanation
    """
    # Ensure instance is 2D
    if len(np.array(instance).shape) == 1:
        instance = np.array([instance])
    
    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(instance.shape[1])]
    
    # Generate explanation
    explanation = explain_model(model, instance, feature_names, class_names, model_type)
    
    # Get prediction
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(instance)[0]
        prediction = model.predict(instance)[0]
    else:
        prediction = model.predict(instance)[0]
        probabilities = None
    
    # Get SHAP values for this instance
    shap_values = explanation['shap_values']
    
    # If multi-class, get SHAP values for predicted class
    if isinstance(shap_values, list) and len(shap_values) > 1:
        if probabilities is not None:
            predicted_class = np.argmax(probabilities)
        else:
            predicted_class = prediction
        instance_shap = shap_values[predicted_class][0]
    else:
        if isinstance(shap_values, list):
            instance_shap = shap_values[0][0]
        else:
            instance_shap = shap_values[0]
    
    # Get feature importance
    feature_importance = [(feature_names[i], instance_shap[i]) for i in range(len(feature_names))]
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Limit to top N features
    top_features = feature_importance[:num_features]
    
    # Create result
    result = {
        'prediction': prediction,
        'probabilities': probabilities,
        'feature_importance': feature_importance,
        'top_features': top_features,
        'shap_values': instance_shap,
        'feature_names': feature_names,
        'instance': instance[0]
    }
    
    return result

def plot_feature_importance(explanation, class_idx=None, max_features=20, plot_type='bar'):
    """
    Plot feature importance based on SHAP values.
    
    Parameters
    ----------
    explanation : dict
        The explanation object from explain_model
    class_idx : int, optional
        Index of the class to explain (for multi-class models)
    max_features : int, optional
        Maximum number of features to show
    plot_type : str, optional
        Type of plot ('bar', 'beeswarm', or 'waterfall')
        
    Returns
    -------
    Figure
        Matplotlib figure with feature importance plot
    """
    shap_values = explanation['shap_values']
    feature_names = explanation['feature_names']
    data = explanation['data']
    
    # Handle multiclass explanations
    if isinstance(shap_values, list) and len(shap_values) > 1:
        if class_idx is None:
            class_idx = 0  # Default to first class
        
        shap_values_to_plot = shap_values[class_idx]
        class_label = f"Class {class_idx}"
        
        if explanation['class_names'] is not None and class_idx < len(explanation['class_names']):
            class_label = explanation['class_names'][class_idx]
    else:
        shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
        class_label = "Output"
    
    plt.figure(figsize=(10, 8))
    
    if plot_type == 'bar':
        # Plot mean absolute SHAP values
        shap_abs_mean = np.abs(shap_values_to_plot).mean(axis=0)
        sorted_idx = np.argsort(shap_abs_mean)[-max_features:]
        
        y_pos = np.arange(len(sorted_idx))
        
        plt.barh(y_pos, shap_abs_mean[sorted_idx])
        plt.yticks(y_pos, [feature_names[i] for i in sorted_idx])
        plt.xlabel(f'Mean |SHAP| value (impact on {class_label})')
        plt.title(f'Feature Importance for {class_label}')
        
    elif plot_type == 'beeswarm':
        # Use SHAP's summary plot
        if isinstance(data, pd.DataFrame):
            features_to_plot = data.iloc[:, :max_features]
        else:
            features_to_plot = data[:, :max_features]
            
        shap.summary_plot(
            shap_values_to_plot, 
            features_to_plot,
            feature_names=feature_names[:max_features],
            max_display=max_features,
            show=False
        )
        plt.title(f'Feature Impact on {class_label}')
        
    elif plot_type == 'waterfall':
        # Use SHAP's waterfall plot for a single example
        if isinstance(data, pd.DataFrame):
            instance = data.iloc[0]
        else:
            instance = data[0]
            
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_to_plot[0], 
                base_values=np.mean(shap_values_to_plot), 
                data=instance,
                feature_names=feature_names
            ),
            max_display=max_features,
            show=False
        )
        plt.title(f'Explanation for {class_label} Prediction')
    
    plt.tight_layout()
    return plt.gcf()

def feature_importance(model, data, target=None, method='shap', n_repeats=10, random_state=42):
    """
    Calculate feature importance using various methods.
    
    Parameters
    ----------
    model : object
        The trained model
    data : array-like
        Input features
    target : array-like, optional
        Target values, required for permutation importance
    method : str, optional
        Method to use ('shap', 'permutation', or 'built-in')
    n_repeats : int, optional
        Number of repeats for permutation importance
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing feature importance values
    """
    if isinstance(data, pd.DataFrame):
        feature_names = data.columns.tolist()
        X = data.values
    else:
        X = data
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    importance = {}
    
    if method == 'shap':
        # Use SHAP values
        explanation = explain_model(model, X, feature_names)
        shap_values = explanation['shap_values']
        
        if isinstance(shap_values, list):
            # For multi-class, take mean across all classes
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
            
        importance['values'] = shap_importance
        
    elif method == 'permutation':
        # Permutation importance
        if target is None:
            raise ValueError("Target is required for permutation importance")
            
        perm_importance = permutation_importance(
            model, X, target, n_repeats=n_repeats, random_state=random_state
        )
        importance['values'] = perm_importance.importances_mean
        importance['std'] = perm_importance.importances_std
        
    elif method == 'built-in':
        # Use model's built-in feature importance
        if hasattr(model, 'feature_importances_'):
            importance['values'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance['values'] = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't have built-in feature importance")
    
    # Create feature importance dict
    feature_imp = {}
    for i, name in enumerate(feature_names):
        feature_imp[name] = importance['values'][i]
    
    importance['features'] = feature_imp
    
    return importance

def explain_misclassifications(model, data, target, class_names=None, threshold=0.7):
    """
    Explain misclassifications to understand why certain samples are misclassified.
    
    Parameters
    ----------
    model : object
        The trained model
    data : array-like
        Input features
    target : array-like
        True target values
    class_names : list, optional
        Names of the target classes
    threshold : float, optional
        Probability threshold for finding confident misclassifications
        
    Returns
    -------
    dict
        Dictionary containing misclassification explanations
    """
    # Convert to numpy arrays if needed
    X = data.values if isinstance(data, pd.DataFrame) else data
    y_true = target
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        max_proba = np.max(y_proba, axis=1)
    else:
        y_pred = model.predict(X)
        max_proba = np.ones_like(y_pred)  # No probabilities available
    
    # Find misclassifications
    misclassified = y_pred != y_true
    
    # Find confident misclassifications (model was confident but wrong)
    confident_misclassified = np.logical_and(misclassified, max_proba >= threshold)
    
    # Get indices
    misclassified_indices = np.where(misclassified)[0]
    confident_indices = np.where(confident_misclassified)[0]
    
    # Create misclassification summary
    summary = {
        'total_samples': len(y_true),
        'misclassified_count': np.sum(misclassified),
        'misclassification_rate': np.mean(misclassified),
        'confident_misclassified_count': np.sum(confident_misclassified),
        'misclassified_indices': misclassified_indices,
        'confident_misclassified_indices': confident_indices
    }
    
    # If we have at least one misclassified sample, explain it
    explanations = []
    if len(confident_indices) > 0:
        # Sort by confidence (most confident mistakes first)
        sorted_indices = confident_indices[np.argsort(-max_proba[confident_indices])]
        
        # Explain top 5 most confident mistakes
        for idx in sorted_indices[:5]:
            instance = X[idx:idx+1]
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            
            # Get class names if available
            if class_names:
                true_class = class_names[true_label]
                pred_class = class_names[pred_label]
            else:
                true_class = f"Class {true_label}"
                pred_class = f"Class {pred_label}"
            
            # Explain this prediction
            explanation = explain_prediction(model, instance)
            
            # Add to explanations
            explanations.append({
                'index': idx,
                'true_label': true_label,
                'predicted_label': pred_label,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': max_proba[idx],
                'feature_importance': explanation['top_features']
            })
    
    summary['explanations'] = explanations
    
    return summary

def generate_explanation_report(explanation, format='text'):
    """
    Generate a comprehensive explanation report.
    
    Parameters
    ----------
    explanation : dict
        The explanation object from explain_model
    format : str, optional
        Report format ('text', 'html', or 'dict')
        
    Returns
    -------
    str or dict
        The explanation report
    """
    if format == 'dict':
        return explanation
    
    shap_values = explanation['shap_values']
    feature_names = explanation['feature_names']
    
    # Get global feature importance
    if isinstance(shap_values, list):
        # Multi-class case
        global_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        global_importance = np.abs(shap_values).mean(axis=0)
    
    # Sort features by importance
    sorted_idx = np.argsort(-global_importance)
    sorted_features = [(feature_names[i], global_importance[i]) for i in sorted_idx]
    
    # Create report
    if format == 'text':
        report = []
        report.append("MODEL EXPLANATION REPORT")
        report.append("=" * 80)
        
        # Global feature importance
        report.append("\nGLOBAL FEATURE IMPORTANCE")
        report.append("-" * 80)
        for i, (feature, importance) in enumerate(sorted_features[:20]):  # Top 20 features
            report.append(f"{i+1}. {feature}: {importance:.4f}")
        
        # Feature impact direction (positive/negative)
        report.append("\nFEATURE IMPACT DIRECTION")
        report.append("-" * 80)
        
        for feature, _ in sorted_features[:10]:  # Top 10 features
            feature_idx = feature_names.index(feature)
            
            # Calculate average impact
            if isinstance(shap_values, list):
                # For multi-class, take mean across all classes
                avg_impact = np.mean([sv[:, feature_idx].mean() for sv in shap_values])
            else:
                avg_impact = shap_values[:, feature_idx].mean()
            
            direction = "Positive" if avg_impact > 0 else "Negative"
            report.append(f"{feature}: {direction} impact (avg = {avg_impact:.4f})")
        
        return "\n".join(report)
    
    elif format == 'html':
        # Simple HTML report
        html = ["<html><body>"]
        html.append("<h1>Model Explanation Report</h1>")
        
        # Global feature importance
        html.append("<h2>Global Feature Importance</h2>")
        html.append("<table border='1'><tr><th>#</th><th>Feature</th><th>Importance</th></tr>")
        
        for i, (feature, importance) in enumerate(sorted_features[:20]):
            html.append(f"<tr><td>{i+1}</td><td>{feature}</td><td>{importance:.4f}</td></tr>")
        
        html.append("</table>")
        html.append("</body></html>")
        
        return "\n".join(html)
    
    return "Unsupported format" 