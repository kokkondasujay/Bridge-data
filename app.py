import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model
@st.cache_resource
def load_model():
    try:
        # Load using joblib (ensure model.pkl is in your HF repo)
        return joblib.load('model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 2. UI Setup
st.set_page_config(page_title="Bridge Assessment", layout="centered")
st.title("🌉 Bridge Condition Predictor")

st.subheader("Enter Bridge Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age of Bridge (years)", min_value=0, value=10)
    traffic = st.number_input("Traffic Volume", min_value=0, value=1000)

with col2:
    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox("Maintenance Level", ["No-Maintainance", "Annual", "Bi-Annual"])

# 3. Manual Encoding (Crucial if not using an sklearn Pipeline)
# Match these numbers to exactly how you labeled them during training!
material_map = {"Concrete": 0, "Steel": 1}
maint_map = {"No-Maintainance": 0, "Annual": 1, "Bi-Annual": 2}

# 4. Prepare Data for Prediction
if st.button("Predict Condition"):
    if model:
        # Create a dataframe with the same column names as your training set X
        input_df = pd.DataFrame([{
            "Age_of_Bridge": age,
            "Traffic_Volume": traffic,
            "Material_Type": material_map[material],
            "Maintenance_Level": maint_map[maintenance]
        }])

        try:
            prediction = model.predict(input_df)
            
            # 5. Display Result
            st.divider()
            if prediction[0] == 0:
                st.success("### Prediction: Good Condition (0)")
            else:
                st.error("### Prediction: Poor Condition (1)")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model file not found. Please check your file upload.")
