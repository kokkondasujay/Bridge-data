import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the model
@st.cache_resource
def load_model():
    # Make sure 'model.pkl' is in the same folder as this script
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
# Temporary debug code
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Actual error: {e}")

# 2. UI Setup
st.set_page_config(page_title="Bridge Condition Predictor", layout="centered")
st.title("🌉 Bridge Condition Assessment")
st.write("Provide the bridge specifications below to predict its condition.")

# 3. Sidebar or Main Page Inputs
st.subheader("Bridge Specifications")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age of Bridge (years)", min_value=0, max_value=200, value=20)
    traffic = st.number_input("Traffic Volume", min_value=0, value=5000)

with col2:
    # Use the exact categories from your training data
    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox("Maintenance Level", ["No-Maintainance", "Annual", "Bi-Annual"])

# 4. Data Preparation
# Creating a dictionary for the input data
input_data = {
    "Age_of_Bridge": age,
    "Traffic_Volume": traffic,
    "Material_Type": material,
    "Maintenance_Level": maintenance
}

# Convert to DataFrame (this ensures column names match your training X)
input_df = pd.DataFrame([input_data])

# 5. Prediction Logic
if st.button("Analyze Condition"):
    try:
        # Note: If your model was trained using a Pipeline (with encoding included), 
        # you can pass the DataFrame directly. 
        # If not, you may need to encode the categories here first.
        prediction = model.predict(input_df)
        
        # Display Result
        result = "Good Condition (0)" if prediction[0] == 0 else "Poor Condition (1)"
        
        if prediction[0] == 0:
            st.success(f"### Prediction: {result}")
        else:
            st.error(f"### Prediction: {result}")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check if your model requires pre-encoded categorical values.")
