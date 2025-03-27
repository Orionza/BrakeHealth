import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("brake_health_dataset_integer.csv")


X = df.drop(columns=["brake_health"])
y = df["brake_health"]


model = GradientBoostingClassifier()
model.fit(X, y)


st.title("Brake Health Prediction")
st.markdown("Enter your vehicle information for the brake pad condition you want to estimate.")

st.markdown("---")


total_km = st.number_input("Total Kilometers")
harsh_braking = st.number_input("Harsh Braking Count")
avg_speed = st.number_input("Average Speed (km/h)")
ignition_duration = st.number_input("Ignition Duration (hours)")
engine_rpm = st.number_input("Engine RPM")
brake_temp = st.number_input("Brake Temperature (°C)")

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "total_km": total_km,
        "harsh_braking": harsh_braking,
        "avg_speed": avg_speed,
        "ignition_duration": ignition_duration,
        "engine_rpm": engine_rpm,
        "brake_temp": brake_temp
    }])

    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Brake Health:  {prediction.upper()}")
