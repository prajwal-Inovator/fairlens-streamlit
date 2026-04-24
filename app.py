import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import time

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Header */
    .main-header {
        font-size: 3.8rem;
        font-weight: 800;
        margin-bottom: 0px;
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-top: 1.5rem;
        letter-spacing: -1.5px;
        animation: fadeInDown 0.8s ease-out;
    }

    .sub-header {
        font-size: 1.25rem;
        color: #a1a1aa;
        margin-bottom: 2.5rem;
        font-weight: 400;
        text-align: center;
        animation: fadeInUp 0.8s ease-out;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 30, 46, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 1s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 198, 255, 0.2);
        border: 1px solid rgba(0, 198, 255, 0.3);
    }

    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 15px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    .success-text { color: #10b981; }
    .warning-text { color: #f59e0b; }
    .danger-text { color: #ef4444; text-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Custom st.info and st.success */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Light mode fallback for metrics */
    @media (prefers-color-scheme: light) {
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
        }
        .main-header {
            background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
            -webkit-background-clip: text;
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

# --- HELPER FUNCTIONS ---
def generate_ai_insight(result, target_col, sensitive_col):
    """Generates a human-readable AI insight based on the bias metrics."""
    groups = result.get('by_group', [])
    if not groups or len(groups) < 2:
        return "Insufficient group data to generate deep AI insights."
    
    # Sort groups by positive outcome rate
    sorted_groups = sorted(groups, key=lambda x: x.get('selection_rate', 0), reverse=True)
    highest = sorted_groups[0]
    lowest = sorted_groups[-1]
    
    rate_high = highest.get('selection_rate', 0)
    rate_low = lowest.get('selection_rate', 0)
    
    ratio = rate_high / rate_low if rate_low > 0 else float('inf')
    
    insight = f"Based on our analysis of the dataset, there is a distinct disparity in the **'{target_col}'** outcomes across different **'{sensitive_col}'** groups.\n\n"
    
    if ratio > 1.05 and ratio != float('inf'):
        insight += f"💡 **Key Finding:** The **{highest['group']}** group is **{ratio:.1f}x more likely** to receive a positive outcome compared to the **{lowest['group']}** group. "
    elif ratio == float('inf'):
        insight += f"💡 **Key Finding:** The **{lowest['group']}** group has a 0% positive outcome rate, highlighting extreme disparity. "
    else:
        insight += f"💡 **Key Finding:** The outcomes are relatively balanced across groups, with minimal disparity. "
        
    insight += "\n\n"
    
    if result['bias_level'] == 'HIGH':
        insight += f"🚨 **Severity:** The model exhibits **High Bias**. The Equalized Odds difference of **{result['equalized_odds_diff']:.2f}** indicates that the model's accuracy varies significantly between groups. Immediate mitigation is strongly recommended."
    elif result['bias_level'] == 'MODERATE':
        insight += f"⚠️ **Severity:** The model exhibits **Moderate Bias**. While not extreme, the demographic parity difference of **{result['demographic_parity_diff']:.2f}** suggests unequal selection rates. Consider applying mitigation constraints."
    else:
        insight += f"✅ **Severity:** The model exhibits **Low Bias**. Metrics are within acceptable fairness thresholds."
        
    return insight

def create_gauge_chart(bias_level, dp_diff):
    """Creates a Plotly Gauge chart for bias severity."""
    value = dp_diff * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': "Bias Severity Score (DP Diff %)", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': "#10b981"}, # Green
                {'range': [10, 20], 'color': "#f59e0b"}, # Yellow
                {'range': [20, 100], 'color': "#ef4444"}  # Red
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def render_metric_card(label, value, level):
    """Renders a custom styled metric card."""
    color_class = "success-text" if level == "LOW" else "warning-text" if level == "MODERATE" else "danger-text"
    if level == "NEUTRAL":
        color_class = ""
        
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    # Initialize session state for demo mode
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False

    # Header
    st.markdown('<div class="main-header">⚖️ FairLens Auditor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-End AI Bias Detection, Deep Explainability, & Automated Mitigation</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3214/3214746.png", width=60) # Placeholder logo icon
        st.title("Navigation & Config")
        
        # Demo Mode Button
        if st.button("🚀 Load Demo Dataset", type="primary", use_container_width=True):
            st.session_state.demo_mode = True
            st.session_state.run_analysis = True
            st.rerun()
            
        st.divider()
        
        data_source = st.radio(
            "📁 1. Choose Data Source:", 
            ["Use Sample Datasets", "Upload Custom Dataset"],
            index=0 if st.session_state.demo_mode else 1,
            help="For testing, use our curated datasets or upload your own CSV."
        )
        
        st.divider()
        df = None
        default_t = None
        default_s = None
        
        if data_source == "Use Sample Datasets":
            dataset_name = st.selectbox("📌 Select a Dataset", list(SAMPLE_DATASETS.keys()))
            file_path = SAMPLE_DATASETS[dataset_name]['file']
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                default_t = SAMPLE_DATASETS[dataset_name]['default_target']
                default_s = SAMPLE_DATASETS[dataset_name]['default_sensitive']
                st.success(f"✅ Loaded {len(df)} rows.")
            else:
                st.error(f"Sample dataset missing at {file_path}")
                
        elif data_source == "Upload Custom Dataset":
            uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully uploaded {len(df)} rows.")

        st.divider()
        st.markdown("### ⚙️ 2. Advanced Mitigation Settings")
        mitigation_method = st.selectbox("Reduction Method", ["ExponentiatedGradient", "GridSearch"])
        mitigation_constraint = st.selectbox("Fairness Constraint", ["DemographicParity", "EqualizedOdds"])

    # Main Content Area
    if df is not None:
        # Sample max 2000 rows for speed
        df = df.sample(n=min(len(df), 2000), random_state=42)
        
        st.markdown("### 🎯 Data Mapping")
        with st.container(border=True):
            colA, colB, colC = st.columns([1, 1, 1])
            
            with colA:
                t_idx = list(df.columns).index(default_t) if default_t in df.columns else 0
                target_col = st.selectbox("🎯 Target Column (What to predict)", df.columns, index=t_idx)
            with colB:
                s_idx = list(df.columns).index(default_s) if default_s in df.columns else 1
                sensitive_col = st.selectbox("🛡️ Sensitive Attribute (e.g. race, gender)", df.columns, index=s_idx)
            with colC:
                st.write("")
                st.write("")
                
                # Validation: Prevent same column selection
                if target_col == sensitive_col:
                    st.error("Target and Sensitive columns cannot be the same!")
                    can_run = False
                else:
                    can_run = True
                    run_btn = st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True, disabled=not can_run)
                    if run_btn:
                        st.session_state.run_analysis = True

            # Preview Selected Columns
            if can_run:
                with st.expander("👀 View Selected Columns Preview", expanded=False):
                    prev_col1, prev_col2 = st.columns(2)
                    with prev_col1:
                        st.markdown(f"**Target: `{target_col}`** Distribution")
                        st.bar_chart(df[target_col].value_counts())
                    with prev_col2:
                        st.markdown(f"**Sensitive: `{sensitive_col}`** Distribution")
                        st.bar_chart(df[sensitive_col].value_counts())
        
        st.divider()

        if st.session_state.run_analysis and can_run:
            tab1, tab2, tab3 = st.tabs(["📊 Bias Detection", "🧠 Explainability", "🛡️ Model Mitigation"])
            
            # Save temp CSV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                df.to_csv(tmp.name, index=False)
                tmp_path = tmp.name
            
            try:
                # ==========================================
                # TAB 1: BIAS DETECTION
                # ==========================================
                with tab1:
                    with st.spinner("🔍 Analyzing dataset for inherent biases..."):
                        time.sleep(0.5) # Slight delay for UI feel
                        result = run_bias_analysis(tmp_path, target_col, sensitive_col)
                        
                    st.markdown("### 🚨 Fairness Report")
                    
                    # Top Metrics
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        render_metric_card("Model Accuracy", f"{result['accuracy'] * 100:.1f}%", "NEUTRAL")
                    with m2:
                        render_metric_card("Demographic Parity Diff", f"{result['demographic_parity_diff']:.3f}", result['bias_level'])
                    with m3:
                        render_metric_card("Equalized Odds Diff", f"{result['equalized_odds_diff']:.3f}", result['bias_level'])
                    
                    col_chart, col_ai = st.columns([1, 1])
                    
                    with col_chart:
                        st.markdown("#### 🌡️ Bias Severity Gauge")
                        fig_gauge = create_gauge_chart(result['bias_level'], result['demographic_parity_diff'])
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col_ai:
                        st.markdown("#### 🤖 AI Fairness Insight")
                        with st.container(border=True):
                            insight_text = generate_ai_insight(result, target_col, sensitive_col)
                            st.markdown(insight_text)
                    
                    st.divider()
                    st.markdown("### 📈 Group Disparities Visualization")
                    
                    group_df = pd.DataFrame(result['by_group'])
                    if not group_df.empty:
                        # Find disadvantaged group (lowest selection rate)
                        lowest_group_idx = group_df['selection_rate'].idxmin()
                        lowest_group = group_df.loc[lowest_group_idx, 'group']
                        
                        # Fix chart coloring: Convert groups to string to ensure Plotly maps them correctly
                        group_df['group_str'] = group_df['group'].astype(str)
                        lowest_group_str = str(lowest_group)
                        
                        # Create custom color map to highlight disadvantaged group in red
                        color_map = {g: '#ff4b4b' if g == lowest_group_str else '#00c6ff' for g in group_df['group_str']}
                        
                        fig = px.bar(
                            group_df, x='group_str', y='selection_rate', color='group_str',
                            title=f"Positive Outcome Rate Across '{sensitive_col}'",
                            labels={'group_str': sensitive_col.capitalize(), 'selection_rate': 'Positive Outcome Rate'},
                            text_auto='.1%',
                            color_discrete_map=color_map
                        )
                        fig.update_traces(marker_line_width=0, opacity=0.9, textfont=dict(color='white'))
                        fig.update_layout(
                            showlegend=False, 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter", size=14),
                            margin=dict(l=20, r=20, t=60, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Note: The **{lowest_group}** group is highlighted in red as it has the lowest positive outcome rate, indicating a potential disadvantage.")
                    else:
                        st.warning("Not enough group data to visualize disparities.")
                
                # ==========================================
                # TAB 2: EXPLAINABILITY
                # ==========================================
                with tab2:
                    with st.spinner("🧠 Generating deep AI explanations via SHAP..."):
                        time.sleep(0.5)
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
                        else:
                            st.warning("Could not generate SHAP explanations for this particular configuration.")
                
                # ==========================================
                # TAB 3: MITIGATION
                # ==========================================
                with tab3:
                    # We use a button here to explicitly trigger mitigation so user sees the change
                    st.markdown("### 🛠️ Fairness Fix")
                    st.write("Apply automated bias mitigation to adjust model behavior and ensure fairer outcomes across all groups.")
                    
                    if st.button("✨ Apply Bias Mitigation", type="primary"):
                        with st.spinner(f"Running {mitigation_method} with {mitigation_constraint}..."):
                            mitigation_result = mitigate_bias(
                                tmp_path, target_col, sensitive_col, 
                                method=mitigation_method, 
                                constraint=mitigation_constraint
                            )
                             
                        st.success(f"**Mitigation Complete!** Strategy Applied: `{mitigation_result['method_used']}` | Constraint: `{mitigation_result['constraint_used']}`.")
                        
                        st.markdown("#### 📊 Performance Comparison")
                        
                        # Comparison Metric Cards
                        col_bef, col_aft = st.columns(2)
                        
                        with col_bef:
                            st.markdown("##### 🚫 Original Sub-Optimal Model")
                            with st.container(border=True):
                                st.metric("Accuracy", f"{mitigation_result['original_accuracy']*100:.1f}%")
                                st.metric("Demographic Parity Diff", f"{mitigation_result['original_dp_diff']:.3f}")
                                st.metric("Equalized Odds Diff", f"{mitigation_result['original_eo_diff']:.3f}")
                            
                        with col_aft:
                            st.markdown("##### ✅ Mitigated Fair Model")
                            with st.container(border=True):
                                acc_delta = mitigation_result['fair_accuracy'] - mitigation_result['original_accuracy']
                                dp_delta = mitigation_result['fair_dp_diff'] - mitigation_result['original_dp_diff']
                                eo_delta = mitigation_result['fair_eo_diff'] - mitigation_result['original_eo_diff']
                                
                                st.metric("Accuracy", f"{mitigation_result['fair_accuracy']*100:.1f}%", f"{acc_delta*100:.1f}%")
                                st.metric("Demographic Parity Diff", f"{mitigation_result['fair_dp_diff']:.3f}", f"{dp_delta:.3f}", delta_color="inverse")
                                st.metric("Equalized Odds Diff", f"{mitigation_result['fair_eo_diff']:.3f}", f"{eo_delta:.3f}", delta_color="inverse")
                            
                        st.info(f"✨ Demographic Parity improved by **{mitigation_result['improvement_dp']:.1f}%**")
                        st.info(f"✨ Equalized Odds improved by **{mitigation_result['improvement_eo']:.1f}%**")
                        
                        # Visual Comparison Chart
                        orig_groups = pd.DataFrame(mitigation_result['original_by_group'])
                        fair_groups = pd.DataFrame(mitigation_result['fair_by_group'])
                        
                        if not orig_groups.empty and not fair_groups.empty:
                            st.markdown("##### 📈 Before / After Selection Rates")
                            
                            orig_groups['Model'] = 'Original'
                            fair_groups['Model'] = 'Mitigated'
                            combined = pd.concat([orig_groups, fair_groups])
                            
                            fig_comp = px.bar(
                                combined, x='group', y='selection_rate', color='Model',
                                barmode='group',
                                title="Positive Outcome Rate Before vs After Mitigation",
                                labels={'selection_rate': 'Positive Outcome Rate', 'group': sensitive_col.capitalize()},
                                color_discrete_map={'Original': '#ef4444', 'Mitigated': '#10b981'},
                                text_auto='.1%'
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)
                
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
    else:
        st.info("👈 Please select a dataset from the sidebar to begin.")

if __name__ == "__main__":
    main()
