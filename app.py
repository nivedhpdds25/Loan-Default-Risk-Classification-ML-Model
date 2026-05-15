import os
import json
import joblib
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. ADVANCED UI & BEAUTIFICATION (CSS)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Loan Risk AI", page_icon="🏦", layout="centered")

def inject_design():
    st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }

    /* Glassmorphism Effect for containers */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }

    /* Professional Headers */
    h1 { color: #1e293b; font-family: 'DM Sans', sans-serif; font-weight: 700; }
    
    /* Result Cards */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        backdrop-filter: blur(5px);
    }
    .risk-high { background: rgba(255, 235, 235, 0.9); border: 2px solid #ef4444; color: #b91c1c; }
    .risk-low { background: rgba(236, 253, 245, 0.9); border: 2px solid #10b981; color: #065f46; }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-2px); }
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
            if f.endswith('.json'):
                with open(f) as j: data[key] = json.load(j)
            else: data[key] = joblib.load(f)
        else: missing.append(f)
    return data, missing

assets, missing_files = load_assets()

# ══════════════════════════════════════════════════════════════════
# 3. SIDEBAR BRANDING
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🏦 LoanRisk AI")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.info("**Nivedh**\nBioinformatics & ML")
    st.markdown("---")
    if not missing_files:
        st.success("Model Status: Online")
    else:
        st.error("Model Status: Offline")

# ══════════════════════════════════════════════════════════════════
# 4. MAIN INTERFACE
# ══════════════════════════════════════════════════════════════════
st.title("Loan Default Risk Analysis")
st.write("Determine creditworthiness using high-precision machine learning.")

if missing_files:
    st.error(f"Required files missing: {', '.join(missing_files)}")
    st.stop()

with st.form("loan_form"):
    st.markdown("### 📄 Applicant Profile")
    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Annual Income ($)", 0, 10000000, 50000)
        credit = st.number_input("Loan Amount ($)", 0, 10000000, 100000)
        annuity = st.number_input("Monthly Installment ($)", 0, 500000, 5000)
    with c2:
        age = st.slider("Age", 18, 90, 35)
        emp = st.slider("Years Employed", 0, 50, 5)
        ext2 = st.slider("Credit Score (Ext 2)", 0.0, 1.0, 0.5)
        ext3 = st.slider("Credit Score (Ext 3)", 0.0, 1.0, 0.5)
    
    submitted = st.form_submit_button("GENERATE RISK REPORT")

if submitted:
    # Feature engineering to match training
    raw = {
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -int(age * 365), "DAYS_EMPLOYED": -int(emp * 365),
        "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "CREDIT_INCOME_RATIO": credit / (income + 1), "ANNUITY_INCOME_RATIO": annuity / (income + 1)
    }
    
    top_feats = assets["top_feats"]
    X = pd.DataFrame([{f: raw.get(f, 0.0) for f in top_feats}])[top_feats]
    
    prob = assets["model"].predict_proba(X)[0][1]
    thresh = assets["threshold"]["threshold"]

    st.markdown("---")
    st.markdown("### 📊 Analysis Result")
    
    # Visual Probability Bar
    st.write(f"Confidence Level: {prob:.1%}")
    st.progress(prob)

    if prob >= thresh:
        st.markdown(f"""<div class="result-card risk-high">
            <h2>⚠️ HIGH RISK</h2>
            <p>The probability of default ({prob:.1%}) exceeds the safety threshold of {thresh:.2%}.</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-card risk-low">
            <h2>✅ APPROVED</h2>
            <p>The probability of default ({prob:.1%}) is within safe limits (Threshold: {thresh:.2%}).</p>
            </div>""", unsafe_allow_html=True)
