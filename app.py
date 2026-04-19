import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add ml-engine to sys.path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml-engine')))

from core.bias_detector import run_bias_analysis
from core.explainer import explain_model
from core.fair_model import mitigate_bias
from core.preprocessor import preprocess

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FairLens | AI Bias Auditor", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Styling for a Hackathon-Ready Look */
    .main-header {
        font-size: 3.2rem;
        font-weight: 900;
        margin-bottom: 0px;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    /* Stylish metric containers */
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Light mode fallback for metrics */
    @media (prefers-color-scheme: light) {
        div[data-testid="metric-container"] {
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
SAMPLE_DATASETS = {
    'Adult Income (Income > 50K)': {
        'file': 'ml-engine/datasets/adult_income.csv',
        'default_target': 'income',
        'default_sensitive': 'sex'
    },
    'German Credit (Credit Risk)': {
        'file': 'ml-engine/datasets/german_credit.csv',
        'default_target': 'credit_risk',
        'default_sensitive': 'sex'
    },
    'COMPAS (Recidivism)': {
        'file': 'ml-engine/datasets/compas.csv',
        'default_target': 'two_year_recid',
        'default_sensitive': 'race'
    }
}

# --- HEADER SECTION ---
st.markdown('<div class="main-header">⚖️ FairLens Auditor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End AI Bias Detection, Deep Explainability, & Automated Mitigation</div>', unsafe_allow_html=True)

# --- SIDEBAR & DATA INGESTION ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    data_source = st.radio(
        "1. Choose Data Source:", 
        ["Use Sample Datasets", "Upload Custom Dataset"],
        help="For testing, use our curated datasets or upload your own CSV."
    )
    
    st.divider()
    df = None
    target_col = None
    sensitive_col = None
    
    if data_source == "Use Sample Datasets":
        dataset_name = st.selectbox("📌 Select a Dataset", list(SAMPLE_DATASETS.keys()))
        file_path = SAMPLE_DATASETS[dataset_name]['file']
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            default_t = SAMPLE_DATASETS[dataset_name]['default_target']
            default_s = SAMPLE_DATASETS[dataset_name]['default_sensitive']
            st.success(f"Loaded {len(df)} rows.")
        else:
            st.error(f"Sample dataset missing at {file_path}")
            
    elif data_source == "Upload Custom Dataset":
        uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            default_t = None
            default_s = None
            st.success(f"Successfully uploaded {len(df)} rows.")

    st.divider()
    st.markdown("### 2. Advanced Mitigation Settings")
    mitigation_method = st.selectbox("Reduction Method", ["ExponentiatedGradient", "GridSearch"])
    mitigation_constraint = st.selectbox("Fairness Constraint", ["DemographicParity", "EqualizedOdds"])

# --- MAIN APP LOGIC ---
if df is not None:
    # Always sample max 2000 rows for speed as requested
    df = df.sample(n=min(len(df), 2000), random_state=42)
    
    # Configuration UI
    st.markdown("### Data Mapping")
    colA, colB, colC = st.columns([1, 1, 1])
    
    with colA:
        t_idx = list(df.columns).index(default_t) if default_t in df.columns else 0
        target_col = st.selectbox("🎯 Target Column (What you want to predict)", df.columns, index=t_idx)
    with colB:
        s_idx = list(df.columns).index(default_s) if default_s in df.columns else 1
        sensitive_col = st.selectbox("🛡️ Sensitive Attribute (e.g. race, gender)", df.columns, index=s_idx)
    with colC:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True)

    # Initial Data Preview
    with st.expander("👀 View Raw Data Preview", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)
    
    st.divider()

    if run_btn:
        # Create tabs for interactive layout
        tab1, tab2, tab3 = st.tabs(["📊 Bias Detection", "🧠 Explainability", "🛡️ Model Mitigation"])
        
        # Save temp CSV for the core functions to process
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            # -------------------------------------------------------------
            # TAB 1: BIAS DETECTION
            # -------------------------------------------------------------
            with tab1:
                with st.spinner("Analyzing dataset for inherent biases..."):
                    result = run_bias_analysis(tmp_path, target_col, sensitive_col)
                    
                st.markdown("### 🚨 Fairness Report")
                
                # Dynamic Alert Box
                if result['bias_level'] == "HIGH":
                    st.error(f"**High Bias Detected:** {result['summary']}")
                elif result['bias_level'] == "MODERATE":
                    st.warning(f"**Moderate Bias Detected:** {result['summary']}")
                else:
                    st.success(f"**Low Bias Detected:** {result['summary']}")
                
                # Core Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Model Accuracy", f"{result['accuracy'] * 100:.2f}%")
                m2.metric("Demographic Parity Diff", f"{result['demographic_parity_diff']:.3f}", delta="Lower is better", delta_color="inverse")
                m3.metric("Equalized Odds Diff", f"{result['equalized_odds_diff']:.3f}", delta="Lower is better", delta_color="inverse")
                
                st.divider()
                st.markdown("### 📈 Group Disparities Visualization")
                
                group_df = pd.DataFrame(result['by_group'])
                if not group_df.empty:
                    # Provide options for visualization
                    vis_metric = st.radio("Select Metric to Visualize:", 
                                         ["Positive Outcome Rate (Selection Rate)", 
                                          "False Positive Rate", 
                                          "False Negative Rate", 
                                          "Accuracy"],
                                         horizontal=True)
                    
                    metric_key_map = {
                        "Positive Outcome Rate (Selection Rate)": "selection_rate",
                        "False Positive Rate": "false_positive_rate",
                        "False Negative Rate": "false_negative_rate",
                        "Accuracy": "accuracy"
                    }
                    selected_col = metric_key_map[vis_metric]
                    
                    fig = px.bar(
                        group_df, x='group', y=selected_col, color='group',
                        title=f"{vis_metric} Across '{sensitive_col}'",
                        labels={'group': sensitive_col.capitalize(), selected_col: vis_metric},
                        text_auto='.1%',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough group data to visualize disparities.")
            
            # -------------------------------------------------------------
            # TAB 2: EXPLAINABILITY (SHAP)
            # -------------------------------------------------------------
            with tab2:
                with st.spinner("Generating deep AI explanations via SHAP..."):
                    df_explain = df.sample(n=min(len(df), 500), random_state=42)
                    X, y, s = preprocess(df_explain, target_col, sensitive_col)
                    
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import train_test_split
                    
                    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                         X, y, s, test_size=0.3, random_state=42
                    )
                    
                    # Explainability model
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    
                    explain_result = explain_model(model, X_train, X_test, X.columns.tolist(), s_test)
                
                st.markdown("### 🧠 AI Feature Importance")
                if 'explanation' in explain_result and 'top_features' in explain_result:
                    st.info(f"**Insight:** {explain_result['explanation']}")
                    
                    top_features = explain_result.get('top_features', [])
                    if top_features and top_features[0]['feature'] != 'error':
                        imp_df = pd.DataFrame(top_features)
                        
                        fig_shap = px.bar(
                            imp_df, x='importance', y='feature', orientation='h', 
                            color='direction',
                            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'},
                            title="Global Feature Impact on Predictions",
                            labels={'importance': 'Mean Absolute SHAP Value', 'feature': 'Feature'}
                        )
                        fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_shap, use_container_width=True)
                        
                        # Add a secondary detailed table if needed
                        with st.expander("Show detailed importance table"):
                             st.dataframe(imp_df.style.format({'importance': '{:.4f}'}))
                    else:
                        st.warning("Could not generate SHAP explanations for this particular configuration.")
            
            # -------------------------------------------------------------
            # TAB 3: MITIGATION
            # -------------------------------------------------------------
            with tab3:
                with st.spinner(f"Running {mitigation_method} with {mitigation_constraint}... (This might take a few seconds)"):
                     mitigation_result = mitigate_bias(
                         tmp_path, target_col, sensitive_col, 
                         method=mitigation_method, 
                         constraint=mitigation_constraint
                     )
                     
                st.markdown("### 🛡️ Automated Fairness Mitigation")
                st.write(f"**Strategy Applied:** `{mitigation_result['method_used']}` using constraint `{mitigation_result['constraint_used']}`.")
                st.success(f"**Result:** {mitigation_result['summary']}")
                
                st.markdown("#### Performance Comparison")
                # Visual Comparison
                col_bef, col_aft = st.columns(2)
                
                with col_bef:
                    st.markdown("##### 🚫 Original Sub-Optimal Model")
                    st.metric("Accuracy", f"{mitigation_result['original_accuracy']*100:.1f}%")
                    st.metric("Demographic Parity Diff", f"{mitigation_result['original_dp_diff']:.3f}")
                    st.metric("Equalized Odds Diff", f"{mitigation_result['original_eo_diff']:.3f}")
                    
                with col_aft:
                    st.markdown("##### ✅ Mitigated Fair Model")
                    acc_delta = mitigation_result['fair_accuracy'] - mitigation_result['original_accuracy']
                    dp_delta = mitigation_result['fair_dp_diff'] - mitigation_result['original_dp_diff']
                    eo_delta = mitigation_result['fair_eo_diff'] - mitigation_result['original_eo_diff']
                    
                    st.metric("Accuracy", f"{mitigation_result['fair_accuracy']*100:.1f}%", f"{acc_delta*100:.1f}%")
                    st.metric("Demographic Parity Diff", f"{mitigation_result['fair_dp_diff']:.3f}", f"{dp_delta:.3f}", delta_color="inverse")
                    st.metric("Equalized Odds Diff", f"{mitigation_result['fair_eo_diff']:.3f}", f"{eo_delta:.3f}", delta_color="inverse")
                    
                st.markdown("#### What changed?")
                st.info(f"✨ Demographic Parity improved by **{mitigation_result['improvement_dp']:.1f}%**")
                st.info(f"✨ Equalized Odds improved by **{mitigation_result['improvement_eo']:.1f}%**")
                
                # Plot differences in groups before vs after
                orig_groups = pd.DataFrame(mitigation_result['original_by_group'])
                fair_groups = pd.DataFrame(mitigation_result['fair_by_group'])
                
                if not orig_groups.empty and not fair_groups.empty:
                    st.markdown("##### Before / After Selection Rates")
                    
                    # Merge data for visualization
                    orig_groups['Model'] = 'Original'
                    fair_groups['Model'] = 'Fair'
                    combined = pd.concat([orig_groups, fair_groups])
                    
                    fig_comp = px.bar(
                        combined, x='group', y='selection_rate', color='Model',
                        barmode='group',
                        title="Positive Outcome Rate Before vs After Mitigation",
                        labels={'selection_rate': 'Positive Outcome Rate', 'group': sensitive_col.capitalize()},
                        color_discrete_map={'Original': '#ef4444', 'Fair': '#10b981'}
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.exception(e)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
else:
    st.info("👈 Please select a dataset from the sidebar to begin.")
