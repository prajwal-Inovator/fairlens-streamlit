import os
import tempfile
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Import core modules
import sys
import os

sys.path.append(os.path.dirname(__file__))

from core.bias_detector import run_bias_analysis
from core.explainer import explain_model
from core.fair_model import mitigate_bias
from core.preprocessor import preprocess

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SAMPLE_DATASETS = {
    'adult_income': {
        'file': 'datasets/adult_income.csv',
        'default_target': 'income',
        'default_sensitive': 'sex'
    },
    'german_credit': {
        'file': 'datasets/german_credit.csv',
        'default_target': 'credit_risk',
        'default_sensitive': 'sex'
    },
    'compas': {
        'file': 'datasets/compas.csv',
        'default_target': 'two_year_recid',
        'default_sensitive': 'race'
    }
}

def save_uploaded_file(file):
    if file.filename == '':
        raise ValueError('No file selected')
    if not file.filename.endswith('.csv'):
        raise ValueError('Only CSV files allowed')

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_path)
    return temp_path

def get_sample_path(name):
    if name not in SAMPLE_DATASETS:
        raise ValueError(f'Unknown dataset: {name}')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, SAMPLE_DATASETS[name]['file'])

@app.route("/")
def home():
    return "FairLens ML Engine Running 🚀"

@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/api/columns', methods=['POST'])
def get_columns():
    try:
        file = request.files['file']
        path = save_uploaded_file(file)

        df = pd.read_csv(path)
        cols = df.columns.tolist()

        os.unlink(path)

        return jsonify({'columns': cols})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        target = request.form.get('targetCol')
        sensitive = request.form.get('sensitiveCol')

        path = save_uploaded_file(file)

        result = run_bias_analysis(path, target, sensitive)

        os.unlink(path)

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain', methods=['POST'])
def explain():
    try:
        file = request.files['file']
        target = request.form.get('targetCol')
        sensitive = request.form.get('sensitiveCol')

        path = save_uploaded_file(file)

        df = pd.read_csv(path)
        X, y, s = preprocess(df, target, sensitive)

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=0.3, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        result = explain_model(model, X_train, X_test, X.columns.tolist(), s_test)

        os.unlink(path)

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/mitigate', methods=['POST'])
def mitigate():
    try:
        file = request.files['file']
        target = request.form.get('targetCol')
        sensitive = request.form.get('sensitiveCol')

        path = save_uploaded_file(file)

        result = mitigate_bias(path, target, sensitive)

        os.unlink(path)

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)