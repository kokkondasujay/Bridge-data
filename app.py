import streamlit as st
import joblib
import pandas as pd
import os

# --- FILE PATH DEBUGGING ---
# This helps if the app says "model.pkl not found"
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')

st.set_page_config(page_title="Bridge Assessment", layout="centered")

# 1. Direct Load with Path Check
if not os.path.exists(model_path):
    st.error(f"❌ Error: 'model.pkl' not found at {model_path}")
    st.info(f"Files currently in folder: {os.listdir(current_dir)}")
    model = None
else:
    try:
        # Load using joblib
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Version Error: {e}")
        st.info("Your model was likely made with an older scikit-learn. Try training it again with the latest version.")
        model = None

# 2. UI Setup
st.title("🌉 Bridge Condition Predictor")

with st.form("bridge_form"):
    st.subheader("Enter Bridge Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age of Bridge (years)", min_value=0, value=10)
        traffic = st.number_input("Traffic Volume", min_value=0, value=1000)
    
    with col2:
        material = st.selectbox("Material Type", ["Concrete", "Steel"])
        maintenance = st.selectbox("Maintenance Level", ["No-Maintainance", "Annual", "Bi-Annual"])
    
    submit = st.form_submit_button("Predict Condition")

# 3. Encoding & Prediction
if submit:
    if model is not None:
        # Map categories to numbers (Must match your training logic!)
        material_map = {"Concrete": 0, "Steel": 1}
        maint_map = {"No-Maintainance": 0, "Annual": 1, "Bi-Annual": 2}
        
        input_df = pd.DataFrame([{
            "Age_of_Bridge": age,
            "Traffic_Volume": traffic,
            "Material_Type": material_map[material],
            "Maintenance_Level": maint_map[maintenance]
        }])

        try:
            prediction = model.predict(input_df)
            st.divider()
            if prediction[0] == 0:
                st.success("### Prediction: Good Condition (0)")
            else:
                st.error("### Prediction: Poor Condition (1)")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("Prediction disabled because model failed to load.")
