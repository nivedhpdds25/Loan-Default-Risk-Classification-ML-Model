import os
import json
import joblib
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. CLEAN INSTITUTIONAL DARK UI
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Loan Risk Analyzer", page_icon="🏦", layout="wide")

def inject_design():
    st.markdown("""
    <style>
    /* Deep Navy Professional Background */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }

    /* Form Container */
    div[data-testid="stForm"] {
        background: #1e293b;
        border-radius: 12px;
        border: 1px solid #334155;
        padding: 40px;
    }

    /* Input text and label visibility */
    label, p, .stNumberInput input {
        color: #e2e8f0 !important;
    }

    /* Result styling */
    .result-card {
        padding: 24px;
        border-radius: 8px;
        text-align: center;
        margin-top: 20px;
        font-weight: 600;
    }
    .risk-high { background: #7f1d1d; border: 1px solid #f87171; color: #fef2f2; }
    .risk-low { background: #064e3b; border: 1px solid #34d399; color: #ecfdf5; }

    /* Button Enhancement */
    .stButton>button {
        background: #2563eb;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    
    .header-text {
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_design()

# ══════════════════════════════════════════════════════════════════
# 2. DATA UTILITIES
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    files = {"model": "lgbm_model.pkl", "scaler": "scaler.pkl", "top_feats": "top_features.pkl", "threshold": "threshold.json"}
    data, missing = {}, []
    for key, f in files.items():
        if os.path.exists(f):
            if f.endswith('.json'):
                with open(f) as j: data[key] = json.load(j)
            else: data[key] = joblib.load(f)
        else: missing.append(f)
    return data, missing

assets, missing_files = load_assets()

# ══════════════════════════════════════════════════════════════════
# 3. INTERFACE
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='header-text'><h1>🏦 Loan Default Risk Classification</h1><p>ML-Powered Credit Risk Assessment Tool</p></div>", unsafe_allow_html=True)

if missing_files:
    st.error(f"Required system files missing: {', '.join(missing_files)}")
    st.stop()

# Layout centering
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    with st.form("loan_analysis_form"):
        st.subheader("Input Financial Profile")
        
        income = st.number_input("Annual Income ($)", 0, 1000000, 60000)
        credit = st.number_input("Total Credit Requested ($)", 0, 2000000, 150000)
        
        col_left, col_right = st.columns(2)
        with col_left:
            age = st.slider("Applicant Age", 18, 85, 30)
            ext_a = st.slider("Internal Risk Score A", 0.0, 1.0, 0.5)
        with col_right:
            employment = st.slider("Years of Employment", 0, 50, 5)
            ext_b = st.slider("Internal Risk Score B", 0.0, 1.0, 0.5)
            
        submitted = st.form_submit_button("RUN CLASSIFICATION")

    if submitted:
        # Preprocessing & Feature Engineering
        # Based on Model_training.ipynb logic
        raw_data = {
            "AMT_INCOME_TOTAL": income, 
            "AMT_CREDIT": credit, 
            "DAYS_BIRTH": -int(age * 365), 
            "DAYS_EMPLOYED": -int(employment * 365),
            "EXT_SOURCE_2": ext_a, 
            "EXT_SOURCE_3": ext_b,
            "CREDIT_INCOME_RATIO": credit / (income + 1)
        }
        
        top_feats = assets["top_feats"]
        # Align input with the top 50 features used in training
        input_df = pd.DataFrame([{f: raw_data.get(f, 0.0) for f in top_feats}])[top_feats]
        
        probability = assets["model"].predict_proba(input_df)[0][1]
        decision_threshold = assets["threshold"]["threshold"]

        st.markdown("---")
        st.write(f"**Default Probability:** {probability:.1%}")
        st.progress(probability)

        if probability >= decision_threshold:
            st.markdown(f"""<div class="result-card risk-high">
                <h3>RESULT: HIGH RISK</h3>
                <p>Classification: LIKELY DEFAULT</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-card risk-low">
                <h3>RESULT: LOW RISK</h3>
                <p>Classification: LIKELY REPAYMENT</p>
                </div>""", unsafe_allow_html=True)
