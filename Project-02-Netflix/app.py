import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and features
model = joblib.load("netflix_final_model.pkl")
feature_names = joblib.load("netflix_features.pkl")

st.title("Netflix Content Type Predictor")

st.write("Predict whether content is Movie or TV Show")

# Input fields
director = st.number_input("Director (encoded)", value=0)
cast = st.number_input("Cast (encoded)", value=0)
release_year = st.number_input("Release Year", 1900, 2025, 2020)
year_added = st.number_input("Year Added", 2000, 2025, 2020)
month_added = st.number_input("Month Added", 1, 12, 6)
duration_num = st.number_input("Duration Number", value=90)

content_age = year_added - release_year

# create full feature vector
input_dict = {}

for feature in feature_names:
    input_dict[feature] = 0

# fill known features
input_dict['director'] = director
input_dict['cast'] = cast
input_dict['release_year'] = release_year
input_dict['year_added'] = year_added
input_dict['month_added'] = month_added
input_dict['duration_num'] = duration_num
input_dict['content_age'] = content_age

# convert to dataframe
input_df = pd.DataFrame([input_dict])

# prediction
if st.button("Predict"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Prediction: Movie 🎬")
    else:
        st.success("Prediction: TV Show 📺")
