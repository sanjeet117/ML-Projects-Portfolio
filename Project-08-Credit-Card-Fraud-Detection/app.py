import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("src/fraud_xgb.pkl","rb"))
scaler = pickle.load(open("src/scaler.pkl","rb"))

st.title("Credit Card Fraud Detection")

amount = st.number_input("Transaction Amount", step=1)
time = st.number_input("Transaction Time", step=1)

if st.button("Predict"):

    # scale amount and time
    scaled = scaler.transform([[amount, time]])
    scaled_amount = scaled[0][0]
    scaled_time = scaled[0][1]

    # create full feature vector (31 features)
    features = np.zeros(30)

    # according to dataset structure
    features[0] = time          # Time
    features[29] = amount       # Amount
    features[30-1] = scaled_amount
    features[31-1] = scaled_time

    prediction = model.predict([features])

    if prediction[0] == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Normal Transaction")