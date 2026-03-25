import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load Model
model = joblib.load('heart_disease_model.pkl')

st.title('src/Heart Disease Prediction App')

age = st.number_input('Age')
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0,1])
cp = st.number_input('Chest Pain Type')
trestbps = st.number_input('Resting Blood Pressure')
chol = st.number_input('Cholesterol')
fbs = st.selectbox('Fasting Blood Sugar > 120 (0/1)', [0,1])
restecg = st.number_input('Resting ECG')
thalach = st.number_input('Max Heart Rate')
exang = st.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0,1])
oldpeak = st.number_input('OldPeak')
slope = st.number_input('Slope')
ca = st.number_input('Number of Major Vessels')
thal = st.number_input('Thal')

if st.button('Predict'):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f'High risk of Heart Disease  Probability: {probability:.2f}')
    else:
        st.success(f'Low risk  Probability: {probability:.2f}')