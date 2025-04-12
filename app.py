import joblib
import pandas as pd
import streamlit as st

# Load trained model
model = joblib.load('gbs_model.pkl')

# Set page title
st.title("Guillain-Barre Syndrome (GBS) Diagnosis Predictor")

# Sidebar info
st.sidebar.write("Provide patient details and nerve test values.")

# Patient info inputs
patient_name = st.text_input("Patient Name:")
gender = st.selectbox("Gender:", options=["Male", "Female", "Other"])
age_input = st.number_input("Age:", min_value=0, max_value=100, value=30)

# Dynamic feature inputs
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

    # Construct nerve table data
    nerve_data = []
    for feature in model_features:
        if feature != "Age":
            nerve_type = " ".join(feature.split()[:-1])
            measure = feature.split()[-1]
            value = input_widgets[feature]

            existing = next((row for row in nerve_data if row["Nerve"] == nerve_type), None)
            if not existing:
                row = {"Nerve": nerve_type, "Latency (ms)": "", "Amplitude (mV)": "", "Conduction Velocity (m/s)": ""}
                nerve_data.append(row)
            else:
                row = existing

            if "Latency" in measure:
                row["Latency (ms)"] = f"{value:.1f} ms"
            elif "Amplitude" in measure:
                row["Amplitude (mV)"] = f"{value:.1f} mV"
            elif "Velocity" in measure:
                row["Conduction Velocity (m/s)"] = f"{value:.1f} m/s"

    # Display report
    st.markdown("<h2 style='text-align: center;'>GBS Report</h2>", unsafe_allow_html=True)
    st.markdown(f"**Patient Name:** {patient_name}")
    st.markdown(f"**Age:** {age_input}")
    st.markdown(f"**Gender:** {gender}")

    st.markdown("### Nerve Conduction Study Results:")
    st.table(pd.DataFrame(nerve_data))

    # Diagnosis
    if prediction == 1:
        st.markdown(
            "<span style='color:red; font-size:18px; font-weight:bold;'>Overall Diagnosis: Positive (Abnormal findings suggest GBS)</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span style='color:green; font-size:18px; font-weight:bold;'>Overall Diagnosis: Negative (Normal findings)</span>",
            unsafe_allow_html=True
        )

    # Confidence
    st.markdown(f"**Prediction Confidence:** ({proba:.2f}% confidence)")

# Footer
st.markdown("---")
st.write("Created for GBS predictive analysis.")

