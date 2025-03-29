"""
Flask application for LlamaEthics dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from llamaethics.bias_detection import detect_bias, fairness_report, plot_attribute_distribution
from llamaethics.explainability import explain_model, plot_feature_importance
from llamaethics.safety_checks import test_robustness, plot_robustness_results

def create_app(config=None):
    """
    Create and configure a Flask application for the LlamaEthics dashboard.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    Flask
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Default configuration
    app.config.update(
        SECRET_KEY='dev',
        UPLOAD_FOLDER='uploads',
        DATASET_FOLDER='datasets',
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max upload
    )
    
    # Update with custom configuration
    if config:
        app.config.update(config)
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Define routes
    @app.route('/')
    def home():
        """Render home page."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Save the file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # Try to load dataset
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                elif file.filename.endswith('.json'):
                    df = pd.read_json(filename)
                else:
                    return jsonify({'error': 'Unsupported file format'}), 400
                
                # Get basic dataset info
                dataset_info = {
                    'filename': file.filename,
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'column_names': df.columns.tolist(),
                    'dtypes': {col: str(df[col].dtype) for col in df.columns}
                }
                
                return jsonify({'success': True, 'dataset_info': dataset_info}), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    @app.route('/analyze_bias', methods=['POST'])
    def analyze_bias():
        """Analyze bias in the dataset."""
        data = request.json
        filename = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        
        # Load dataset
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
            elif filename.endswith('.json'):
                df = pd.read_json(filename)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
                
            # Analyze bias
            sensitive_attr = data['sensitive_attribute']
            target_attr = data['target_attribute']
            
            bias_metrics = detect_bias(df, sensitive_attr, target_attr)
            report = fairness_report(bias_metrics)
            
            # Generate attribute distribution plot
            fig = plot_attribute_distribution(df, sensitive_attr, target_attr)
            
            # Convert plot to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            
            # Create interactive plots with plotly
            plotly_figs = {}
            
            # Demographic parity plot
            dp = bias_metrics['demographic_parity']
            dp_fig = px.bar(
                x=list(dp['positive_rates'].keys()),
                y=list(dp['positive_rates'].values()),
                title=f"Positive Prediction Rate by {sensitive_attr}",
                labels={'x': sensitive_attr, 'y': 'Positive Rate'}
            )
            plotly_figs['demographic_parity'] = pio.to_json(dp_fig)
            
            # Equal opportunity plot
            eo = bias_metrics['equal_opportunity']
            eo_fig = px.bar(
                x=list(eo['true_positive_rates'].keys()),
                y=list(eo['true_positive_rates'].values()),
                title=f"True Positive Rate by {sensitive_attr}",
                labels={'x': sensitive_attr, 'y': 'True Positive Rate'}
            )
            plotly_figs['equal_opportunity'] = pio.to_json(eo_fig)
            
            return jsonify({
                'success': True, 
                'bias_metrics': bias_metrics,
                'report': report,
                'plot_data': plot_data,
                'plotly_figs': plotly_figs
            }), 200
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/explain_model', methods=['POST'])
    def explain_model_endpoint():
        """Explain model predictions."""
        # This would require a trained model
        # For the dashboard demo, we'd need to either:
        # 1. Train a model on the fly
        # 2. Load a pre-trained model
        # 3. Use a mock model for demonstration
        
        return jsonify({'message': 'Model explanation functionality coming soon'}), 200
    
    @app.route('/test_robustness', methods=['POST'])
    def test_robustness_endpoint():
        """Test model robustness."""
        # Similar to explain_model_endpoint, this requires a model
        
        return jsonify({'message': 'Robustness testing functionality coming soon'}), 200
    
    @app.route('/static/<path:path>')
    def serve_static(path):
        """Serve static files."""
        return send_from_directory('static', path)
    
    return app

def run_app(host='0.0.0.0', port=5000, debug=True, config=None):
    """
    Run the LlamaEthics dashboard application.
    
    Parameters
    ----------
    host : str, optional
        Host to run the server on
    port : int, optional
        Port to run the server on
    debug : bool, optional
        Whether to run in debug mode
    config : dict, optional
        Configuration dictionary
    """
    app = create_app(config)
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_app() 