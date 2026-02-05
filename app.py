import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BridgeGuard AI",
    page_icon="🌉",
    layout="centered"
)

# --- PROFESSIONAL CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #0d6efd; color: white; font-weight: bold;
        border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #0b5ed7; border: none; }
    .result-card { 
        padding: 30px; border-radius: 15px; text-align: center; 
        margin-top: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_bridge_model():
    # Ensure model.pkl is in your Hugging Face repo
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_bridge_model()

# --- HEADER SECTION ---
st.title("🌉 Bridge Condition Assessment")
st.markdown("##### AI-Powered Structural Health Monitoring System")
st.write("Provide structural parameters to generate a real-time integrity report.")

# --- INPUT SECTION ---
with st.container():
    st.markdown("### 📝 Infrastructure Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("📅 Age of Bridge (Years)", min_value=0, max_value=250, value=20)
        traffic = st.number_input("🚗 Daily Traffic Volume", min_value=0, value=5000)
    
    with col2:
        # Match these options exactly to your bridge.ipynb training labels
        material = st.selectbox("🏗️ Material Type", ["Concrete", "Steel"])
        maintenance = st.selectbox("🛠️ Maintenance Level", ["Annual", "Bi-Annual", "No-Maintainance"])

# --- PREDICTION LOGIC ---
if st.button("RUN DIAGNOSTIC ANALYSIS"):
    # THE FIX: Create a DataFrame with EXACT column names from your notebook
    # If your notebook used different names, update them here.
    input_data = pd.DataFrame([{
        "Age_of_Bridge": age,
        "Traffic_Volume": traffic,
        "Material_Type": material,
        "Maintenance_Level": maintenance
    }])

    try:
        # The model pipeline handles scaling/encoding automatically via the DataFrame
        prediction = model.predict(input_data)
        
        st.divider()
        st.subheader("📊 Assessment Result")
        
        # Professional Output Cards
        if prediction[0] == 1:
            st.balloons()
            st.markdown(f"""
                <div class="result-card" style="background-color: #d1e7dd; border-left: 10px solid #198754;">
                    <h2 style="color: #0f5132;">✅ STRUCTURAL STATUS: STABLE</h2>
                    <p style="color: #0f5132; font-size: 1.1em;">
                        The analysis indicates the bridge is in <b>Good Condition</b>.<br>
                        Continue with the standard maintenance cycle.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card" style="background-color: #f8d7da; border-left: 10px solid #dc3545;">
                    <h2 style="color: #842029;">⚠️ ALERT: CRITICAL CONDITION</h2>
                    <p style="color: #842029; font-size: 1.1em;">
                        The analysis indicates the bridge is in <b>Poor Condition</b>.<br>
                        <b>Action Required:</b> Immediate physical structural inspection recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if column names in app.py match your model's training data.")
