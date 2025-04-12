import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load trained model
model = joblib.load('gbs_model.pkl')

# Set page title
st.title("Guillain-Barre Syndrome (GBS) Diagnosis Predictor")

# Sidebar info
st.sidebar.write("Provide patient details and nerve test values.")

# Patient information
st.subheader("Patient Information")
patient_name = st.text_input("Patient Name:")
gender = st.selectbox("Gender:", options=["Male", "Female", "Other"])
age_input = st.number_input("Age:", min_value=0, max_value=100, value=30)

# Nerve conduction test values
st.subheader("Nerve Conduction Study Values")
model_features = model.named_steps['scaler'].feature_names_in_
input_widgets = {}

for feature in model_features:
    if feature != 'Age':
        input_widgets[feature] = st.number_input(f"{feature}:", value=0.0)

# Predict button
if st.button("Predict GBS"):
    input_values = [age_input] + [input_widgets[col] for col in model_features if col != 'Age']
    input_data = pd.DataFrame([input_values], columns=model_features)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction] * 100
    result = "Positive" if prediction == 1 else "Negative"

    st.markdown("### 🧾 Prediction Result")
    st.write(f"**Patient Name:** {patient_name}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age_input}")
    st.success(f"Diagnosis: {result}")
    st.info(f"Prediction Confidence: {proba:.2f}%")

# Footer
st.markdown("---")
st.write("Created for GBS predictive analysis.")
