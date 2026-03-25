import streamlit as st
import pickle
import pandas as pd

# Load saved objects
model = pickle.load(open("churn_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer details")

# Inputs

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=18.0,
    max_value=120.0,
    value=70.0,
    step=0.01,   # decimal fix
    format="%.2f"
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=9000.0,
    value=1000.0,
    step=0.01,   # decimal fix
    format="%.2f"
)

senior = st.selectbox("Senior Citizen",[0,1])
partner = st.selectbox("Partner",[0,1])
dependents = st.selectbox("Dependents",[0,1])

# Prediction

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[tenure, monthly, total, senior, partner, dependents]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "SeniorCitizen",
            "Partner",
            "Dependents"
        ]
    )

    # match training columns
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # scale
    input_scaled = scaler.transform(input_data)

    # prediction
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.write("Churn Probability:", round(prob,3))

    if pred == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")