import os
import json
import joblib
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. UI DESIGN WITH PERSISTENT BENTO HEIGHT
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Loan Risk Dashboard", page_icon="🏦", layout="wide")

def inject_design():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        background-image: 
            linear-gradient(rgba(30, 41, 59, 0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 41, 59, 0.4) 1px, transparent 1px);
        background-size: 35px 35px;
        color: #f1f5f9;
    }

    /* Forced Minimum Height for Bento Cards */
    .bento-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 20px;
        min-height: 580px; /* Forces both sides to look similar in height */
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }

    h1, h2, h3 {
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    div[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
    
    /* Result Box Styling */
    .res-box { padding: 25px; border-radius: 12px; text-align: center; margin-top: 20px; }
    .risk-high { background: rgba(127, 29, 29, 0.5); border: 1px solid #ef4444; color: #fecaca; }
    .risk-low { background: rgba(6, 78, 59, 0.5); border: 1px solid #10b981; color: #d1fae5; }
    
    .meta-label { color: #94a3b8; font-size: 0.85rem; margin-top: 10px; }
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
            try:
                if f.endswith('.json'):
                    with open(f) as j: data[key] = json.load(j)
                else: data[key] = joblib.load(f)
            except: missing.append(f)
        else: missing.append(f)
    return data, missing

assets, missing_files = load_assets()

# ══════════════════════════════════════════════════════════════════
# 3. DASHBOARD INTERFACE
# ══════════════════════════════════════════════════════════════════
st.markdown("<h1 style='text-align: center;'>🏦 Loan Default Risk AI</h1>", unsafe_allow_html=True)

if missing_files:
    st.error(f"Missing Files: {', '.join(missing_files)}")
    st.stop()

col_left, col_right = st.columns([1.2, 1], gap="medium")

with col_left:
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    with st.form("bento_form"):
        st.subheader("📋 Application Details")
        
        income = st.number_input("Annual Income ($)", 0, 5000000, 65000)
        credit = st.number_input("Loan Amount Requested ($)", 0, 5000000, 150000)
        
        c1, c2 = st.columns(2)
        age = c1.slider("Applicant Age", 18, 85, 32)
        emp = c2.slider("Years Employed", 0, 50, 7)
        
        st.markdown("---")
        st.write("External Financial Scoring")
        ext_a = st.slider("Agency Score A", 0.0, 1.0, 0.51)
        ext_b = st.slider("Agency Score B", 0.0, 1.0, 0.49)
        
        submitted = st.form_submit_button("RUN RISK CLASSIFICATION")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.subheader("🎯 Analysis Report")
    
    if not submitted:
        # This section fills the "Blank Space" before submission
        st.markdown("""
        <div style='margin-top: 20px;'>
            <p style='color: #94a3b8;'>Awaiting input data. Once submitted, the system will process 
            the profile against 50 engineered features.</p>
            <hr style='border-color: rgba(255,255,255,0.1);'>
            <p class='meta-label'>MODEL ARCHITECTURE</p>
            <code style='color: #60a5fa;'>LightGBM Classifier</code>
            <p class='meta-label'>DECISION THRESHOLD</p>
            <code>Optimized for Recall (0.70+)</code>
            <p class='meta-label'>PIPELINE STATUS</p>
            <span style='color: #10b981;'>● Ready for Inference</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Preprocessing
        raw = {
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, 
            "DAYS_BIRTH": -int(age * 365), "DAYS_EMPLOYED": -int(emp * 365),
            "EXT_SOURCE_2": ext_a, "EXT_SOURCE_3": ext_b,
            "CREDIT_INCOME_RATIO": credit / (income + 1)
        }
        top_feats = assets["top_feats"]
        X = pd.DataFrame([{f: raw.get(f, 0.0) for f in top_feats}])[top_feats]
        
        prob = assets["model"].predict_proba(X)[0][1]
        thresh = assets["threshold"]["threshold"]

        st.write(f"Default Probability Score: **{prob:.1% Rose}**")
        st.progress(prob)

        if prob >= thresh:
            st.markdown(f"""<div class="res-box risk-high">
                <h3>RESULT: REJECTED</h3>
                <p>High default risk detected ({prob:.1%})</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="res-box risk-low">
                <h3>RESULT: APPROVED</h3>
                <p>Applicant meets safety criteria ({prob:.1%})</p>
                </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
