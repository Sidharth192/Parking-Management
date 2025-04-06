import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.express as px
import os

# Page config
st.set_page_config(page_title="Parking Predictor", page_icon="🅿️", layout="centered")

# Load model
model_path = "parking_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file 'parking_model.pkl' not found.")
    st.stop()

model = joblib.load(model_path)

# Title
st.title("🅿️ Parking Availability & Price Predictor")
st.markdown("Get predictions for parking availability and estimated pricing for the next few hours.")

# --- User Inputs ---
st.subheader("🗕️ Select Entry Time")

selected_date = st.date_input("Choose a date", value=datetime.date.today())
selected_time = st.time_input("Choose a time (8:00 AM - 11:00 PM)", value=datetime.time(8, 0))

# Time validation
if selected_time < datetime.time(8, 0) or selected_time > datetime.time(23, 0):
    st.warning("⚠️ Please choose a time between 8:00 AM and 11:00 PM.")
    st.stop()

selected_datetime = datetime.datetime.combine(selected_date, selected_time)
hour = selected_datetime.hour
day = selected_datetime.weekday()
month = selected_datetime.month

# --- Predict Availability ---
features = pd.DataFrame([[hour, day, month]], columns=["hour", "day_of_week", "month"])
availability = model.predict(features)[0]
availability_label = "🟢 OPEN" if availability == 1 else "🔴 FULL"

# --- Predict Next Available Slot ---
next_available_time = None
if availability == 0:
    for i in range(1, 13):
        future_dt = selected_datetime + datetime.timedelta(hours=i)
        ffeatures = pd.DataFrame([[future_dt.hour, future_dt.weekday(), future_dt.month]], columns=["hour", "day_of_week", "month"])
        if model.predict(ffeatures)[0] == 1:
            next_available_time = future_dt.strftime("%I:%M %p")
            break

# --- Price Logic ---
def calculate_price(hour, day, availability):
    base_price = 100
    price = base_price
    if 8 <= hour <= 11 or 17 <= hour <= 20:
        price += 20
    if day >= 5:
        price += 15
    if availability == 0:
        price += 30
    return price

predicted_price = calculate_price(hour, day, availability)

# --- Output ---
st.markdown("### 📊 Prediction Results")
col1, col2 = st.columns(2)
with col1:
    st.metric("Availability", availability_label)
    if next_available_time:
        st.caption(f"⏰ Next available slot at: {next_available_time}")
with col2:
    st.metric("Estimated Price", f"₹{predicted_price}")

# --- Pricing Trend ---
st.markdown("### 📈 Pricing Forecast (Next 4 Hours)")

future_predictions = []
for i in range(4):  # current hour + 3 next (restricted to 4 hours total)
    future_dt = selected_datetime + datetime.timedelta(hours=i)
    fhour = future_dt.hour
    fday = future_dt.weekday()
    fmonth = future_dt.month

    ffeatures = pd.DataFrame([[fhour, fday, fmonth]], columns=["hour", "day_of_week", "month"])
    favail = model.predict(ffeatures)[0]
    fprice = calculate_price(fhour, fday, favail)

    future_predictions.append({
        "Time": future_dt.strftime("%I:%M %p"),
        "Predicted Price (₹)": fprice
    })

future_df = pd.DataFrame(future_predictions)

fig = px.line(
    future_df,
    x="Time",
    y="Predicted Price (₹)",
    markers=True,
    title="🕒 Price Forecast for Selected Slot",
    line_shape="spline",
)
fig.update_traces(line=dict(color="blue", width=3), marker=dict(size=8))
fig.update_layout(title_x=0.2)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(future_df, use_container_width=True)

st.markdown("---")
st.caption("🚗 car parking price and availability prediction using random forest classifier built for AIOT internship at DIGITOAD by Students of VVCE")
