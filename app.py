import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. UI SETUP & CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Loan Risk Predictor", page_icon="🏦", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1a56db; color: white; }
    .risk-high { background: #fff0f0; border: 2px solid #e53e3e; border-radius: 10px; padding: 20px; text-align: center; color: #c53030; font-weight: bold; }
    .risk-low { background: #f0fff4; border: 2px solid #38a169; border-radius: 10px; padding: 20px; text-align: center; color: #276749; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 2. MODEL LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    files = {
        "model": "lgbm_model.pkl",
        "scaler": "scaler.pkl",
        "top_feats": "top_features.pkl",
        "threshold": "threshold.json"
    }
    data = {}
    missing = []
    for key, f in files.items():
        if os.path.exists(f):
            if f.endswith('.json'):
                with open(f) as j: data[key] = json.load(j)
            else:
                data[key] = joblib.load(f)
        else:
            missing.append(f)
    return data, missing

assets, missing_files = load_assets()

# ══════════════════════════════════════════════════════════════════
# 3. APP INTERFACE
# ══════════════════════════════════════════════════════════════════
st.title("🏦 Loan Default Risk Predictor")

if missing_files:
    st.error(f"Missing required files in repo: {', '.join(missing_files)}")
    st.stop()

st.info("Enter applicant details below to calculate the probability of default.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        credit = st.number_input("Total Loan Amount ($)", min_value=0, value=100000, step=5000)
        annuity = st.number_input("Monthly Payment (Annuity)", min_value=0, value=5000)
        
    with col2:
        age = st.slider("Age (Years)", 18, 90, 35)
        employed_yrs = st.slider("Years Employed", 0, 50, 5)
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5, 0.01)
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5, 0.01)

    submit = st.form_submit_button("Run Risk Analysis")

if submit:
    # 1. Feature Engineering (Matching your training logic)
    # Note: Days are usually negative in this specific dataset (Home Credit)
    raw_input = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -int(age * 365),
        "DAYS_EMPLOYED": -int(employed_yrs * 365),
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "CREDIT_INCOME_RATIO": credit / (income + 1),
        "ANNUITY_INCOME_RATIO": annuity / (income + 1)
    }

    # 2. Align with Top 50 features
    top_feats = assets["top_feats"]
    input_row = {f: 0.0 for f in top_feats}
    for k, v in raw_input.items():
        if k in input_row:
            input_row[k] = v
            
    X_df = pd.DataFrame([input_row])[top_feats]
    
    # 3. Prediction
    prob = assets["model"].predict_proba(X_df)[0][1]
    thresh = assets["threshold"]["threshold"]

    # 4. Results Display
    st.divider()
    if prob >= thresh:
        st.markdown(f'<div class="risk-high">⚠️ HIGH RISK<br>Default Probability: {prob:.1%}</div>', unsafe_allow_html=True)
        st.error(f"Probability is above the decision threshold of {thresh:.2f}.")
    else:
        st.markdown(f'<div class="risk-low">✅ LOW RISK<br>Default Probability: {prob:.1%}</div>', unsafe_allow_html=True)
        st.success(f"Probability is below the decision threshold of {thresh:.2f}.")

    with st.expander("Technical Breakdown"):
        st.write(f"**Model Type:** LightGBM Classifier")
        st.write(f"**Decision Threshold:** {thresh}")
        st.write(f"**Raw Probability:** {prob:.4f}")