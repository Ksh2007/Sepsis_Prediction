from pathlib import Path
import streamlit as st
import requests

st.set_page_config(page_title="Sepsis Prediction System", page_icon="🏥")

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
IMAGE_PATH = PROJECT_ROOT / 'images' / "sepsis_infographic.png"

try:
    if IMAGE_PATH.exists():
        # Updated from use_column_width to use_container_width
        st.image(str(IMAGE_PATH), use_container_width=True)
    else:
        st.caption(f"ℹ️ Layout note: Infographic image file not found at: {IMAGE_PATH}")
except Exception as img_err:
    st.caption("ℹ️ Layout note: Image rendering skipped to prevent interface locking.")

st.title("🏥 Sepsis Prediction System")
st.markdown("Enter the patient's clinical metrics below to predict the likelihood of sepsis.")

st.subheader("Patient Clinical Metrics")
col1, col2 = st.columns(2)

with col1:
    prg = st.number_input("PRG (Pregnancies / Plasma Glucose index)", min_value=0, max_value=20, value=0)
    pl = st.number_input("PL (Plasma Glucose Concentration)", min_value=0.0, max_value=300.0, value=120.0)
    pr = st.number_input("PR (Blood Pressure mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    sk = st.number_input("SK (Triceps Skin Fold Thickness mm)", min_value=0.0, max_value=100.0, value=20.0)

with col2:
    ts = st.number_input("TS (2-Hour Serum Insulin mu U/ml)", min_value=0, max_value=1000, value=80)
    m11 = st.number_input("M11 (Body Mass Index kg/m²)", min_value=0.0, max_value=70.0, value=32.0)
    bd2 = st.number_input("BD2 (Diabetes Pedigree Function)", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=33)

if st.button("Predict Sepsis Status", type="primary"):
    payload = {
        "PRG": int(prg), "PL": float(pl), "PR": float(pr), "SK": float(sk),
        "TS": int(ts), "M11": float(m11), "BD2": float(bd2), "Age": int(age)
    }
    
    try:
        with st.spinner("Analyzing metrics against the machine learning model..."):
            # Ensure the correct port and endpoint matching your FastAPI backend
            response = requests.post("http://127.0.0.1:7860/predict/", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            sepsis_status = result.get("Sepsis", "Unknown")
            
            st.write("---")
            if sepsis_status.lower() in ["positive", "1", "yes"]:
                st.error(f"### 🚨 Prediction: Sepsis Detected ({sepsis_status})")
                st.markdown("**Recommendation:** Immediate clinical evaluation and monitoring are strongly advised.")
            else:
                st.success(f"### 🎉 Prediction: No Sepsis Detected ({sepsis_status})")
                st.markdown("**Note:** Patient metrics fall within expected stable bounds.")
        else:
            st.error(f"❌ Backend Server Error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Could not connect to the backend server: {e}")