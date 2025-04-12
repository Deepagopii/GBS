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

# Patient details
patient_name = st.text_input("Patient Name:")
gender = st.selectbox("Gender:", options=["Male", "Female", "Other"])
age_input = st.number_input("Age:", min_value=0, max_value=100, value=30)

# Input features
model_features = model.named_steps['scaler'].feature_names_in_
input_widgets = {}

st.subheader("Enter Nerve Conduction Values:")
for feature in model_features:
    if feature != 'Age':
        input_widgets[feature] = st.number_input(f"{feature}:", value=0.0)

# Predict button
if st.button("Predict GBS"):
    # Prepare input for prediction
    input_values = [age_input] + [input_widgets[col] for col in model_features if col != 'Age']
    input_data = pd.DataFrame([input_values], columns=model_features)

    # Prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction] * 100
    result = "Positive" if prediction == 1 else "Negative"

    # Report display
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>GBS Report</h2>", unsafe_allow_html=True)

    st.markdown(f"**Patient Name:** {patient_name}<br>**Age:** {age_input}<br>**Gender:** {gender}", unsafe_allow_html=True)
    st.markdown("<br><strong>Nerve Conduction Study Results:</strong><br>", unsafe_allow_html=True)

    # Build table data
    nerve_data = {}
    for feature in model_features:
        if feature == "Age":
            continue
        parts = feature.split()
        if "Conduction" in feature:
            nerve = " ".join(parts[:-2])  # e.g., 'Ulnar Motor'
            measure = "Conduction Velocity (m/s)"
        else:
            nerve = " ".join(parts[:-1])
            last_word = parts[-1]
            if "Latency" in last_word:
                measure = "Latency (ms)"
            elif "Amplitude" in last_word:
                measure = "Amplitude (mV)"
            else:
                continue

        if nerve not in nerve_data:
            nerve_data[nerve] = {
                "Nerve": nerve,
                "Latency (ms)": "",
                "Amplitude (mV)": "",
                "Conduction Velocity (m/s)": ""
            }

        val = input_widgets[feature]
        if "Latency" in measure:
            nerve_data[nerve]["Latency (ms)"] = f"{val:.1f} ms"
        elif "Amplitude" in measure:
            nerve_data[nerve]["Amplitude (mV)"] = f"{val:.1f} mV"
        elif "Conduction" in measure:
            nerve_data[nerve]["Conduction Velocity (m/s)"] = f"{val:.1f} m/s"

    nerve_df = pd.DataFrame(nerve_data.values())
    st.table(nerve_df)

    # Diagnosis section
    if prediction == 1:
        st.markdown(f"<h4 style='color:red;'>Overall Diagnosis: Positive (Abnormal findings suggest GBS)</h4>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4 style='color:green;'>Overall Diagnosis: Negative (Normal findings)</h4>", unsafe_allow_html=True)

    st.markdown(f"<b>Prediction Confidence:</b> ({proba:.2f}% confidence)", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Created for GBS predictive analysis.")


