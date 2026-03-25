import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- LOAD MODEL ----------------
model_path = os.path.join(os.path.dirname(__file__), "hotel_model.pkl")
model = pickle.load(open(model_path, "rb"))

st.title("Hotel Booking Cancellation Predictor")

st.write("Fill all details below:")

# ---------------- NUMERIC INPUTS ----------------
lead_time = st.number_input("Lead Time", min_value=0, step=1)
arrival_year = st.number_input("Arrival Year", min_value=2015, max_value=2025, step=1)
arrival_week = st.number_input("Arrival Week Number", min_value=1, max_value=53, step=1)
arrival_day = st.number_input("Arrival Day", min_value=1, max_value=31, step=1)

stays_weekend = st.number_input("Weekend Nights", min_value=0, step=1)
stays_week = st.number_input("Week Nights", min_value=0, step=1)

adults = st.number_input("Adults", min_value=0, step=1)
children = st.number_input("Children", min_value=0, step=1)
babies = st.number_input("Babies", min_value=0, step=1)

previous_cancellations = st.number_input("Previous Cancellations", min_value=0, step=1)
previous_not_cancel = st.number_input("Previous Not Canceled", min_value=0, step=1)

booking_changes = st.number_input("Booking Changes", min_value=0, step=1)
waiting_days = st.number_input("Waiting Days", min_value=0, step=1)

adr = st.number_input("ADR (Average Daily Rate)", min_value=0, step=1)

parking = st.number_input("Parking Spaces", min_value=0, step=1)
special_requests = st.number_input("Special Requests", min_value=0, step=1)

is_repeated = st.selectbox("Repeated Guest", [0, 1])

# ---------------- CATEGORICAL INPUTS ----------------
hotel = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
month = st.selectbox("Arrival Month", [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
])

meal = st.selectbox("Meal", ["BB","HB","FB","SC"])
country = st.text_input("Country Code (e.g. IND, PRT)", "IND")

market_segment = st.selectbox("Market Segment", [
    "Direct","Corporate","Online TA","Offline TA/TO","Complementary","Groups"
])

distribution_channel = st.selectbox("Distribution Channel", [
    "Direct","Corporate","TA/TO","GDS"
])

reserved_room = st.selectbox("Reserved Room Type", list("ABCDEFG"))
assigned_room = st.selectbox("Assigned Room Type", list("ABCDEFG"))

deposit_type = st.selectbox("Deposit Type", [
    "No Deposit","Refundable","Non Refund"
])

customer_type = st.selectbox("Customer Type", [
    "Transient","Contract","Transient-Party","Group"
])

# ---------------- PREDICT ----------------
if st.button("Predict"):

    data = pd.DataFrame([{
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": month,
        "arrival_date_week_number": arrival_week,
        "arrival_date_day_of_month": arrival_day,
        "stays_in_weekend_nights": stays_weekend,
        "stays_in_week_nights": stays_week,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_not_cancel,
        "reserved_room_type": reserved_room,
        "assigned_room_type": assigned_room,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "days_in_waiting_list": waiting_days,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special_requests
    }])

    # -------- FEATURE ENGINEERING --------
    data["total_guest"] = data["adults"] + data["children"] + data["babies"]
    data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data["booking_change_flag"] = data["booking_changes"].apply(lambda x: 1 if x > 0 else 0)
    data["waiting_flag"] = data["days_in_waiting_list"].apply(lambda x: 1 if x > 0 else 0)

    # -------- DROP SAME AS TRAINING --------
    data.drop([
        "adults","children","babies",
        "stays_in_week_nights",
        "days_in_waiting_list",
        "booking_changes"
    ], axis=1, inplace=True)

    # -------- PREDICTION --------
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("Booking will be CANCELED")
    else:
        st.success("Booking will NOT be canceled")