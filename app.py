import os
import json
import joblib
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. QUAD-BENTO UI DESIGN (No more huge chunks!)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Loan Risk Dashboard", page_icon="🏦", layout="wide")

def inject_design():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(#1e293b 1px, transparent 1px);
        background-size: 30px 30px;
        color: #f1f5f9;
    }
    /* Small, clean cards instead of huge chunks */
    .bento-tile {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    h3 {
        color: #60a5fa;
        font-size: 1.1rem !important;
        margin-bottom: 15px !important;
    }
    /* Styling result boxes */
    .res-box { padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; }
    .risk-high { background: rgba(153, 27, 27, 0.6); border: 1px solid #f87171; }
    .risk-low { background: rgba(20, 83, 45, 0.6); border: 1px solid #34d399; }
    
    /* Center the button */
    .stButton>button { width: 100%; background: #2563eb; color: white; border: none; height: 3rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

inject_design()

# ══════════════════════════════════════════════════════════════════
# 2. ASSET LOADING
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
# 3. INTERFACE
# ══════════════════════════════════════════════════════════════════
st.markdown("<h2 style='text-align: center; color: white;'>🏦 Loan Default Classifier</h2>", unsafe_allow_html=True)

if missing_files:
    st.error(f"Missing Files: {', '.join(missing_files)}")
    st.stop()

# Using a form to wrap everything for one-click submission
with st.form("main_form"):
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        # Tile 1: Personal
        st.markdown('<div class="bento-tile">', unsafe_allow_html=True)
        st.markdown("### 👤 Personal Profile")
        age = st.slider("Applicant Age", 18, 85, 30)
        emp = st.slider("Employment Tenure (Years)", 0, 50, 5)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tile 2: Financial
        st.markdown('<div class="bento-tile">', unsafe_allow_html=True)
        st.markdown("### 💰 Financial Data")
        income = st.number_input("Annual Income ($)", 0, 1000000, 60000)
        credit = st.number_input("Requested Loan Amount ($)", 0, 2000000, 150000)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Tile 3: Scores
        st.markdown('<div class="bento-tile">', unsafe_allow_html=True)
        st.markdown("### 📈 Credit Scoring")
        scr_a = st.slider("Internal Score A", 0.0, 1.0, 0.5)
        scr_b = st.slider("Internal Score B", 0.0, 1.0, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tile 4: Action & Result
        st.markdown('<div class="bento-tile">', unsafe_allow_html=True)
        st.markdown("### 🎯 Final Decision")
        submitted = st.form_submit_button("GENERATE REPORT")
        
        if submitted:
            # Preprocessing
            raw = {"AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "DAYS_BIRTH": -int(age * 365), 
                   "DAYS_EMPLOYED": -int(emp * 365), "EXT_SOURCE_2": scr_a, "EXT_SOURCE_3": scr_b,
                   "CREDIT_INCOME_RATIO": credit / (income + 1)}
            
            top_feats = assets["top_feats"]
            X = pd.DataFrame([{f: raw.get(f, 0.0) for f in top_feats}])[top_feats]
            
            prob = assets["model"].predict_proba(X)[0][1]
            thresh = assets["threshold"]["threshold"]

            st.write(f"Default Probability: **{prob:.1%}**")
            st.progress(prob)
            
            if prob >= thresh:
                st.markdown(f'<div class="res-box risk-high">REJECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="res-box risk-low">APPROVED</div>', unsafe_allow_html=True)
        else:
            st.info("Fill out the data and click the button to see the results.")
        st.markdown('</div>', unsafe_allow_html=True)
