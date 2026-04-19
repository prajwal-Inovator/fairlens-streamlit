import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add ml-engine to sys.path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml-engine')))

from core.bias_detector import run_bias_analysis
from core.explainer import explain_model
from core.fair_model import mitigate_bias
from core.preprocessor import preprocess

# Streamlit Page Config
st.set_page_config(page_title="FairLens - AI Bias Auditor", page_icon="🔍", layout="wide")

# Constants
SAMPLE_DATASETS = {
    'Adult Income': {
        'file': 'ml-engine/datasets/adult_income.csv',
        'default_target': 'income',
        'default_sensitive': 'sex'
    },
    'German Credit': {
        'file': 'ml-engine/datasets/german_credit.csv',
        'default_target': 'credit_risk',
        'default_sensitive': 'sex'
    },
    'COMPAS': {
        'file': 'ml-engine/datasets/compas.csv',
        'default_target': 'two_year_recid',
        'default_sensitive': 'race'
    }
}

st.title("FairLens - AI Bias Auditor 🔍")
st.markdown("A unified tool to detect, explain, and mitigate biases in your machine learning datasets.")

st.sidebar.header("Data Configuration")
data_source = st.sidebar.radio("Select Data Source", ["Sample Dataset", "Upload CSV"])

df = None
target_col = None
sensitive_col = None

if data_source == "Sample Dataset":
    dataset_name = st.sidebar.selectbox("Choose a Sample Dataset", list(SAMPLE_DATASETS.keys()))
    file_path = SAMPLE_DATASETS[dataset_name]['file']
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        default_t = SAMPLE_DATASETS[dataset_name]['default_target']
        default_s = SAMPLE_DATASETS[dataset_name]['default_sensitive']
    else:
        st.error(f"Sample dataset not found at {file_path}")
        
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        default_t = None
        default_s = None

if df is not None:
    # Sample max 2000 rows for speed as requested
    df = df.sample(n=min(len(df), 2000), random_state=42)
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        # Determine index for default selection
        t_idx = list(df.columns).index(default_t) if default_t in df.columns else 0
        target_col = st.selectbox("Select Target Column", df.columns, index=t_idx)
    with col2:
        s_idx = list(df.columns).index(default_s) if default_s in df.columns else 1
        sensitive_col = st.selectbox("Select Sensitive Column", df.columns, index=s_idx)

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running Bias Analysis..."):
            # Save temp CSV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                df.to_csv(tmp.name, index=False)
                tmp_path = tmp.name
            
            try:
                # 1. Bias Analysis
                st.header("📊 Bias Analysis Results")
                result = run_bias_analysis(tmp_path, target_col, sensitive_col)
                
                # Display Results
                st.success(f"Analysis Complete! Bias Level: {result['bias_level']}")
                st.markdown(f"**Summary:** {result['summary']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{result['accuracy']:.4f}")
                m2.metric("Demographic Parity Diff", f"{result['demographic_parity_diff']:.4f}")
                m3.metric("Equalized Odds Diff", f"{result['equalized_odds_diff']:.4f}")
                
                st.subheader("Group Details")
                group_df = pd.DataFrame(result['by_group'])
                st.dataframe(group_df)
                
                st.divider()
                
                # 2. Explainability
                st.header("🧠 Explainability")
                with st.spinner("Generating Explanations..."):
                     # Sample max 500 rows for explainability
                     df_explain = df.sample(n=min(len(df), 500), random_state=42)
                     
                     X, y, s = preprocess(df_explain, target_col, sensitive_col)
                     from sklearn.linear_model import LogisticRegression
                     from sklearn.model_selection import train_test_split
                     
                     X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                         X, y, s, test_size=0.3, random_state=42
                     )
                     
                     model = LogisticRegression(max_iter=1000)
                     model.fit(X_train, y_train)
                     
                     explain_result = explain_model(model, X_train, X_test, X.columns.tolist(), s_test)
                     
                     # Simple rendering of importance
                     if 'global_importance' in explain_result:
                         st.subheader("Global Feature Importance")
                         imp_df = pd.DataFrame({
                             'Feature': explain_result['global_importance']['features'],
                             'Importance': explain_result['global_importance']['values']
                         })
                         st.bar_chart(imp_df.set_index('Feature'))
                
                st.divider()
                
                # 3. Mitigation
                st.header("🛡️ Mitigation")
                with st.spinner("Mitigating Bias..."):
                     mitigation_result = mitigate_bias(tmp_path, target_col, sensitive_col)
                     
                     st.write("Before Mitigation:")
                     b1, b2 = st.columns(2)
                     b1.metric("Accuracy", f"{mitigation_result['original_accuracy']:.4f}")
                     b2.metric("Demographic Parity Diff", f"{mitigation_result['original_dp_diff']:.4f}")

                     st.write("After Mitigation:")
                     a1, a2 = st.columns(2)
                     a1.metric("Accuracy", f"{mitigation_result['fair_accuracy']:.4f}", f"{mitigation_result['fair_accuracy'] - mitigation_result['original_accuracy']:.4f}")
                     a2.metric("Demographic Parity Diff", f"{mitigation_result['fair_dp_diff']:.4f}", f"{mitigation_result['original_dp_diff'] - mitigation_result['fair_dp_diff']:.4f}")                     
                     
                     st.write(f"**Improvement:** DP improved by {mitigation_result['improvement_dp']:.1f}%, EO improved by {mitigation_result['improvement_eo']:.1f}%")
                     st.write(f"**Summary:** {mitigation_result['summary']}")
                     st.success("Bias Mitigation completed successfully!")
            
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
