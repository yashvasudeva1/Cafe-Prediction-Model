import streamlit as st
import joblib
import numpy as np

# Load the trained model and polynomial feature transformer
model = joblib.load("polynomial_model.pkl")
poly = joblib.load("poly_transformer.pkl")

st.set_page_config(page_title="Cafe Total Spent Predictor", layout="centered")

st.title("Cafe Total Spent Predictor")

st.markdown("Enter the order details below:")

# User Inputs
price_per_unit = st.number_input("Price Per Unit (₹)", value=50.0)
quantity = st.number_input("Quantity", value=2)
day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=7)

# When user clicks the predict button
if st.button("Predict Total Spent"):
    input_data = np.array([[price_per_unit, quantity, day, month]])
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)[0]
    st.success(f"✅ Predicted Total Spent: ₹{prediction:.2f}")
