import streamlit as st
import joblib
import numpy as np

model = joblib.load('linear_model.pkl')

st.title("Cafe Sales Predictor")

st.markdown("Enter the transaction details below:")

item = st.selectbox("Item", ["Coffee", "Tea", "Sandwich", "Cake"])
units_sold = st.number_input("Units Sold", min_value=1, value=1)
payment_method = st.selectbox("Payment Method", ["Cash", "Card", "UPI"])
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
promo = st.radio("Promo Applied?", ["Yes", "No"])

item_map = {"Coffee": 0, "Tea": 1, "Sandwich": 2, "Cake": 3}
payment_map = {"Cash": 0, "Card": 1, "UPI": 2}
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
promo_map = {"No": 0, "Yes": 1}

features = np.array([[item_map[item], units_sold, payment_map[payment_method], day_map[day_of_week], promo_map[promo]]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Total Spent: ₹{prediction:.2f}")
