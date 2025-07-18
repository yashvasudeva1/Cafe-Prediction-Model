import streamlit as st
import joblib
import numpy as np
poly = joblib.load("poly_features.pkl")
model = joblib.load("polynomial_model.pkl")
mapping for internal feature generation (not shown to user)
item_price_dict = {
    "Cake": 6.0,
    "Smoothie": 4.5,
    "Coffee": 3.0,
    "Salad": 5.0,
    "Cookie": 2.0,
    "Tea": 2.5,
    "Juice": 3.5,
    "Sandwich": 5.5
}
st.title("Cafe Total Spent Predictor (Polynomial Regression)")
st.markdown("Select item and enter transaction details below:")
item_options = [
    "Cake", "Smoothie", "Coffee", "Salad", 
    "Cookie", "Tea", "Juice", "Sandwich"
]
item = st.selectbox("Select Item", options=item_options)
quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1)

if st.button("Predict Total Spent"):
    price_per_unit = item_price_dict[item]
    features = np.array([[price_per_unit, quantity, day, month]])
    features_poly = poly.transform(features)
    prediction = model.predict(features_poly)[0]
    st.success(f"Estimated Total Spent: ${prediction:.2f}")
