import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("src/rf_walmart_model.pkl","rb"))

st.title("Walmart Weekly Sales Forecast")

st.write("Enter previous sales values (example: 45000000)")

# Lag inputs
lag1 = st.number_input("Sales Last Week", min_value=0, step=100000)
lag2 = st.number_input("Sales 2 Weeks Ago", min_value=0, step=100000)
lag3 = st.number_input("Sales 3 Weeks Ago", min_value=0, step=100000)
lag4 = st.number_input("Sales 4 Weeks Ago", min_value=0, step=100000)

lag52 = st.number_input("Sales Same Week Last Year", min_value=0, step=100000)

# Month dropdown
month = st.selectbox(
    "Month",
    list(range(1,13))
)

# Week dropdown
week = st.selectbox(
    "Week Number",
    list(range(1,53))
)

if st.button("Forecast Sales"):

    rolling_mean_4 = np.mean([lag1,lag2,lag3,lag4])
    rolling_mean_12 = rolling_mean_4
    rolling_std_4 = np.std([lag1,lag2,lag3,lag4])

    input_df = pd.DataFrame([[

        2024,
        month,
        week,
        1,
        1,
        1,
        lag1,
        lag2,
        lag3,
        lag4,
        lag52,
        rolling_mean_4,
        rolling_mean_12,
        rolling_std_4

    ]],

    columns=[
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'quarter',
        'lag_1',
        'lag_2',
        'lag_3',
        'lag_4',
        'lag_52',
        'rolling_mean_4',
        'rolling_mean_12',
        'rolling_std_4'
    ])

    prediction = model.predict(input_df)

    st.success(f"Forecasted Weekly Sales: {prediction[0]:,.0f}")