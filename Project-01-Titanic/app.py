import streamlit as st
import pickle
import numpy as np


# Load trained model and scaler

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival")

st.markdown("---")


# User Inputs

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex_input = st.selectbox("Sex", ["female", "male"])
sex = 1 if sex_input == "male" else 0   # SAME as LabelEncoder

age = st.slider("Age", 1, 80, 30)

fare = st.slider("Fare", 0.0, 600.0, 50.0)

family_size = st.slider("Family Size", 1, 10, 1)

embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0
# C is baseline

title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
title_Miss = 1 if title == "Miss" else 0
title_Mr   = 1 if title == "Mr" else 0
title_Mrs  = 1 if title == "Mrs" else 0
title_Rare = 1 if title == "Rare" else 0
# Master is baseline


# Feature Vector (ORDER MATTERS)

features = np.array([[ 
    pclass,          # Pclass
    sex,             # Sex
    age,             # Age
    fare,            # Fare
    family_size,     # FamilySize
    embarked_Q,      # Embarked_Q
    embarked_S,      # Embarked_S
    title_Miss,      # Title_Miss
    title_Mr,        # Title_Mr
    title_Mrs,       # Title_Mrs
    title_Rare       # Title_Rare
]])

# Scale features
features_scaled = scaler.transform(features)

# Prediction

if st.button("Predict Survival"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.success(f"Passenger is likely to **SURVIVE**")
    else:
        st.error(f"Passenger is likely to **NOT SURVIVE**")

    st.info(f"Survival Probability: **{probability:.2%}**")



 