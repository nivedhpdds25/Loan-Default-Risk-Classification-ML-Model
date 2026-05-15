import os
import json
import joblib
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. BENTO BOX & GLASSMORPHISM UI
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Loan Risk Dashboard", page_icon="🏦", layout="wide")

def inject_design():
    st.markdown("""
    <style>
    /* Technical Grid Background */
    .stApp {
        background-color: #0f172a;
        background-image: 
            linear-gradient(rgba(30, 41, 59, 0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 41, 59, 0.4) 1px, transparent 1px);
        background-size: 35px 35px;
        color: #f1f5f9;
    }

    /* Bento Box Card Style */
    .bento-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* Gradient Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Hide standard Streamlit elements for a cleaner look */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Result Cards */
    .res-box { padding: 20px; border-radius: 12px; text-align: center; }
    .risk-high { background: rgba(127, 29, 29, 0.4); border: 1px solid #ef4444; color: #fecaca; }
    .risk-low { background: rgba(6, 78, 59, 0.4); border: 1px solid #10b981; color: #d1fae5; }
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
st.markdown("<h1 style='text-align: center;'>🏦 Loan Default Risk AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Precision ML for Credit Assessment</p>", unsafe_allow_html=True)

if missing_files:
    st.error(f"Missing Assets: {', '.join(missing_files)}")
    st.stop()

# Bento Layout using columns
col_form, col_res = st.columns([1.5, 1], gap="large")

with col_form:
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    with st.form("bento_form"):
        st.subheader("📋 Applicant Data")
        
        # Row 1
        c1, c2 = st.columns(2)
        income = c1.number_input("Annual Income ($)", 0, 1000000, 65000)
        credit = c2.number_input("Loan Amount ($)", 0, 2000000, 150000)
        
        # Row 2
        c3, c4 = st.columns(2)
        age = c3.slider("Age (Years)", 18, 85, 30)
        emp = c4.slider("Work History (Years)", 0, 50, 8)
        
        # Row 3
        st.markdown("---")
        ext_a = st.slider("Credit Risk Factor A", 0.0, 1.0, 0.52)
        ext_b = st.slider("Credit Risk Factor B", 0.0, 1.0, 0.48)
        
        submitted = st.form_submit_button("GENERATE ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    st.markdown('<div class="bento-card" style="height: 100%;">', unsafe_allow_html=True)
    st.subheader("🎯 Live Report")
    
    if not submitted:
        st.info("Waiting for data submission...")
        st.markdown("""
        <small style='color: #64748b;'>
        • Model: LightGBM<br>
        • Feature Count: 50<br>
        • Status: Ready
        </small>
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

        # Results Display
        st.write(f"Default Probability: **{prob:.1%}**")
        st.progress(prob)

        if prob >= thresh:
            st.markdown(f"""<div class="res-box risk-high">
                <h3>RESULT: REJECTED</h3>
                <p>Probability exceeds threshold ({thresh:.2f})</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="res-box risk-low">
                <h3>RESULT: APPROVED</h3>
                <p>Applicant fits safety parameters</p>
                </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
