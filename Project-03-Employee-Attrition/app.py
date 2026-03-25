import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ======================
# Load model and features
# ======================

model = joblib.load("final_attrition_model.pkl")
feature_columns = joblib.load("model_features.pkl")


# ======================
# App Title
# ======================

st.title("Employee Attrition Prediction App")

st.write("Enter employee details to predict attrition")


# ======================
# User Inputs
# ======================

age = st.number_input("Age", min_value=18, max_value=60, value=30)

monthlyincome = st.number_input(
    "MonthlyIncome",
    min_value=1009,
    max_value=19999,
    value=5000
)

jobsatisfaction = st.selectbox(
    "JobSatisfaction",
    [1, 2, 3, 4]
)

yearsatcompany = st.number_input(
    "YearsAtCompany",
    min_value=0,
    max_value=40,
    value=5
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)


# ======================
# Create input dataframe
# ======================

input_data = {
    "Age": age,
    "MonthlyIncome": monthlyincome,
    "JobSatisfaction": jobsatisfaction,
    "YearsAtCompany": yearsatcompany,
    "OverTime": overtime
}

input_df = pd.DataFrame([input_data])


# ======================
# Apply get_dummies
# ======================

input_df = pd.get_dummies(input_df)


# ======================
# Align features with training
# ======================

input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)


# ======================
# Prediction Button
# ======================

if st.button("Predict"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:

        st.error("Employee is likely to leave")

    else:

        st.success(" Employee is likely to stay")

        