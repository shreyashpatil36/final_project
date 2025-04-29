import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved models and dataset
xgb_model = joblib.load("xgb_model.joblib")
dataset = joblib.load("insurance_dataset.joblib")

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
st.title("💰 Insurance Cost Prediction App")
st.markdown("Enter details below to estimate medical insurance charges.")

# Input fields
age = st.slider("Age", 18, 100, 30)
sex = st.radio("Sex", ["Male", "Female"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.radio("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Convert inputs
sex_val = 0 if sex == "Male" else 1
smoker_val = 0 if smoker == "Yes" else 1
region_map = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
region_val = region_map[region]

input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

# Load models
regressor = joblib.load("linear_model.joblib") if "linear_model.joblib" in joblib.os.listdir() else None
rf_model = joblib.load("rf_model.joblib") if "rf_model.joblib" in joblib.os.listdir() else None

# Predict
if st.button("Predict Insurance Cost"):
    xgb_pred = xgb_model.predict(input_data)[0]
    st.subheader(f"🧮 Predicted Cost (XGBoost): ₹ {round(xgb_pred, 2)}")
    
    if regressor:
        st.write(f"Linear Regression: ₹ {round(regressor.predict(input_data)[0], 2)}")
    if rf_model:
        st.write(f"Random Forest: ₹ {round(rf_model.predict(input_data)[0], 2)}")

    # SHAP Explanation
    explainer = shap.Explainer(xgb_model, dataset.drop("charges", axis=1))
    shap_values = explainer(input_data)
    
    st.subheader("🔍 Feature Importance (SHAP)")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=6, show=False)
    st.pyplot(fig)
