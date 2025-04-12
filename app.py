pip install joblib

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Load dataset
df = pd.read_csv('Final__GBS_peripheral_nerves_Dataset.csv')

# Encode 'Diagnosis' (target)
df['Diagnosis'] = df['Diagnosis'].map({'Positive': 1, 'Negative': 0})

# Handle missing Age values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Define nerves and features
features = ['Latency', 'Amplitude', 'Conduction Velocity']
nerves = ['Ulnar', 'Median']
types = ['Motor', 'Sensory']

# Extract and rename features
for nerve in nerves:
    for n_type in types:
        for feature in features:
            mask = (df['Nerve'] == nerve) & (df['Nerve Type'] == n_type)
            df[f"{nerve} {n_type} {feature}"] = np.where(mask, df[feature], np.nan)

# Drop unnecessary columns
df.drop(columns=['Patient ID', 'Latency', 'Amplitude', 'Conduction Velocity', 'Nerve Type', 'Nerve'], inplace=True)

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Define features and target
X = df.select_dtypes(include=np.number).drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression model
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
])
model.fit(X_train, y_train)

# Save the trained model for later use
joblib.dump(model, 'gbs_model.pkl')

# Evaluate Model Performance
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

st.write(f"Model Evaluation:")
st.write(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}")
st.write(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")

# Streamlit UI Elements
name_input = st.text_input("Name:")
gender_input = st.selectbox("Gender:", ["Male", "Female", "Other"])
age_input = st.number_input("Age:", min_value=0, max_value=100)

input_widgets = {}
for feature in X.columns:
    if feature != 'Age':
        input_widgets[feature] = st.number_input(f"{feature}:", value=0.0)

if st.button("Predict GBS"):
    input_values = [age_input] + [input_widgets[col] for col in X.columns if col != 'Age']
    
    # Make prediction using the loaded model
    model = joblib.load('gbs_model.pkl')
    input_data = pd.DataFrame([input_values], columns=X.columns)
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction] * 100
    result = "Positive" if prediction == 1 else "Negative"
    
    st.write(f"Diagnosis: {result}")
    st.write(f"Prediction Confidence: {proba:.2f}%")

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Load dataset
df = pd.read_csv('Final__GBS_peripheral_nerves_Dataset.csv')

# Encode 'Diagnosis' (target)
df['Diagnosis'] = df['Diagnosis'].map({'Positive': 1, 'Negative': 0})

# Handle missing Age values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Define nerves and features
features = ['Latency', 'Amplitude', 'Conduction Velocity']
nerves = ['Ulnar', 'Median']
types = ['Motor', 'Sensory']

# Extract and rename features
for nerve in nerves:
    for n_type in types:
        for feature in features:
            mask = (df['Nerve'] == nerve) & (df['Nerve Type'] == n_type)
            df[f"{nerve} {n_type} {feature}"] = np.where(mask, df[feature], np.nan)

# Drop unnecessary columns
df.drop(columns=['Patient ID', 'Latency', 'Amplitude', 'Conduction Velocity', 'Nerve Type', 'Nerve'], inplace=True)

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Define features and target
X = df.select_dtypes(include=np.number).drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression model
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
])
model.fit(X_train, y_train)

# Save the trained model for later use
joblib.dump(model, 'gbs_model.pkl')

# Evaluate Model Performance
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

st.write(f"Model Evaluation:")
st.write(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}")
st.write(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")

# Streamlit UI Elements
name_input = st.text_input("Name:")
gender_input = st.selectbox("Gender:", ["Male", "Female", "Other"])
age_input = st.number_input("Age:", min_value=0, max_value=100)

input_widgets = {}
for feature in X.columns:
    if feature != 'Age':
        input_widgets[feature] = st.number_input(f"{feature}:", value=0.0)

if st.button("Predict GBS"):
    input_values = [age_input] + [input_widgets[col] for col in X.columns if col != 'Age']
    
    # Make prediction using the loaded model
    model = joblib.load('gbs_model.pkl')
    input_data = pd.DataFrame([input_values], columns=X.columns)
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction] * 100
    result = "Positive" if prediction == 1 else "Negative"
    
    st.write(f"Diagnosis: {result}")
    st.write(f"Prediction Confidence: {proba:.2f}%")
