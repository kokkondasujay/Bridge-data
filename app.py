import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
import os

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

@st.cache_resource
def load_model():
    try:
        if not os.path.exists('model.pkl'):
            st.error("❌ model.pkl not found in repo root")
            st.info("Upload model.pkl via GitHub or drag-drop to Space")
            return None
            
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {str(e)[:200]}...")
        st.info("💡 Retrain model with scikit-learn==1.6.1 or use joblib.dump()")
        return None

# Load model first
model = load_model()
if model is None:
    st.stop()

# UI
st.set_page_config(page_title="Bridge Condition Predictor", layout="wide")
st.title("🌉 Bridge Condition Assessment")
st.markdown("Enter bridge specs below for condition prediction.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", 0, 200, 20)
    traffic = st.number_input("Traffic Volume", 0, 100000, 5000)
with col2:
    material = st.selectbox("Material", ["Concrete", "Steel"])
    maintenance = st.selectbox("Maintenance", ["No-Maintainance", "Annual", "Bi-Annual"])

# Predict
if st.button("🔍 Analyze Condition", type="primary"):
    try:
        input_data = {
            "Age_of_Bridge": age,
            "Traffic_Volume": traffic,
            "Material_Type": material,
            "Maintenance_Level": maintenance
        }
        input_df = pd.DataFrame([input_data])
        
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
        
        col1, col2 = st.columns([3,1])
        with col1:
            status = "🟢 Good Condition" if pred == 0 else "🔴 Poor Condition"
            st.markdown(f"### **{status}**")
        with col2:
            st.metric("Prediction Score", f"{pred}")
            
        if prob is not None:
            st.info(f"**Probabilities:** Good: {prob[0]:.1%} | Poor: {prob[1]:.1%}")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Check if input columns match training data exactly")
