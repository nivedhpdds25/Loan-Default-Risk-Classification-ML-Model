import os
import json
import joblib
import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. BENTO BOX & TECHNICAL UI DESIGN
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Loan Risk Dashboard", page_icon="🏦", layout="wide")

def inject_design():
    st.markdown("""
    <style>
    /* Technical Grid Background with Glow */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(30, 58, 138, 0.2), transparent 40%),
            linear-gradient(rgba(30, 41, 59, 0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 41, 59, 0.4) 1px, transparent 1px);
        background-size: 100% 100%, 35px 35px, 35px 35px;
        color: #f1f5f9;
    }

    /* Bento Card Style */
    .bento-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }

    /* Gradient Typography */
    h1, h2, h3 {
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Form and Input Clean-up */
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
    label, p { color: #cbd5e1 !important; font-weight: 500; }

    /* Result Indicators */
    .res-box { padding: 20px; border-radius: 12px; text-align: center; margin-top: 15px; }
    .risk-high { background: rgba(127, 29, 29, 0.4); border: 1px solid #ef4444; color: #fecaca; }
    .risk-low { background: rgba(6, 78, 59, 0.4); border: 1px solid #10b981; color: #d1fae5; }
    </style>
    """, unsafe_allow_html=True)

inject_design()

# ══════════════════════════════════════════════════════════════════
# 2. BULLETPROOF ASSET LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    # Define files required based on your repository
    files = {
        "model": "lgbm_model.pkl", 
        "scaler": "scaler.pkl", 
        "top_feats": "top_features.pkl", 
        "threshold": "threshold.json"
    }
    data, missing_or_corrupt = {}, []
    
    for key, f in files.items():
        if os.path.exists(f):
            try:
                # Attempt to load assets while catching corruption errors
                if f.endswith('.json'):
                    with open(f) as j: data[key] = json.load(j)
                else:
                    data[key] = joblib.load(f)
            except Exception as e:
                st.error(f"Failed to read {f}: File may be corrupted or empty.")
                missing_or_corrupt.append(f)
        else:
            missing_or_corrupt.append(f)
            
    return data, missing_or_corrupt

assets, missing_files = load_assets()

# ══════════════════════════════════════════════════════════════════
# 3. INTERFACE & BENTO LAYOUT
# ══════════════════════════════════════════════════════════════════
st.markdown("<h1 style='text-align: center;'>🏦 Loan Default Risk AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Industrial-Grade Credit Assessment</p>", unsafe_allow_html=True)

if missing_files:
    st.error(f"System Error: The following files are missing or inaccessible: {', '.join(missing_files)}")
    st.info("Ensure all model files are in the root directory of your GitHub repository.")
    st.stop()

# Dashboard split
col_input, col_output = st.columns([1.4, 1], gap="medium")

with col_input:
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    with st.form("bento_analysis_form"):
        st.subheader("📋 Application Profile")
        
        c1, c2 = st.columns(2)
        income = c1.number_input("Annual Income ($)", 0, 5000000, 65000)
        credit = c2.number_input("Loan Amount ($)", 0, 5000000, 150000)
        
        c3, c4 = st.columns(2)
        age = c3.slider("Age (Years)", 18, 80, 32)
        emp = c4.slider("Tenure (Years)", 0, 50, 6)
        
        st.markdown("---")
        st.caption("Internal Risk Scoring")
        scr_a = st.slider("Agency Score A", 0.0, 1.0, 0.50)
        scr_b = st.slider("Agency Score B", 0.0, 1.0, 0.50)
        
        submitted = st.form_submit_button("GENERATE REPORT")
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="bento-card" style="min-height: 485px;">', unsafe_allow_html=True)
    st.subheader("🎯 Analysis Output")
    
    if not submitted:
        st.write("Awaiting submission...")
        st.caption("Fill out the applicant profile to run the classifier.")
    else:
        # Preprocessing & Feature Alignment
        raw = {
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, 
            "DAYS_BIRTH": -int(age * 365), "DAYS_EMPLOYED": -int(emp * 365),
            "EXT_SOURCE_2": scr_a, "EXT_SOURCE_3": scr_b,
            "CREDIT_INCOME_RATIO": credit / (income + 1)
        }
        
        top_feats = assets["top_feats"]
        # Match input row to exact model expectations
        X = pd.DataFrame([{f: raw.get(f, 0.0) for f in top_feats}])[top_feats]
        
        # Prediction
        prob = assets["model"].predict_proba(X)[0][1]
        threshold = assets["threshold"]["threshold"]

        # Results Visualization
        st.write(f"Risk Probability: **{prob:.1%}**")
        st.progress(prob)

        if prob >= threshold:
            st.markdown(f"""<div class="res-box risk-high">
                <h3>REJECTED</h3>
                <p>Calculated risk exceeds security threshold ({threshold:.2f}).</p>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="res-box risk-low">
                <h3>APPROVED</h3>
                <p>Applicant meets the required creditworthiness criteria.</p>
                </div>""", unsafe_allow_html=True)
                
    st.markdown('</div>', unsafe_allow_html=True)
