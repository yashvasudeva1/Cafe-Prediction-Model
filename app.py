import streamlit as st
import joblib
import numpy as np
poly = joblib.load("poly_features.pkl")
model = joblib.load("polynomial_model.pkl")
item_price_dict = {
    "Coffee": 3.0,
    "Tea": 2.5,
    "Sandwich": 5.0,
    "Muffin": 4.0,
    "Juice": 3.5
}
st.title("Cafe Total Spent Predictor (Polynomial Regression)")
st.markdown("Select item and transaction details:")
item = st.select_slider(
    "Select Item", 
    options=list(item_price_dict.keys()), 
    value=list(item_price_dict.keys())[0]
)
price_per_unit = item_price_dict[item]
st.write(f"**Price per Unit:** ${price_per_unit:.2f}")
quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
if st.button("Predict Total Spent"):
    features = np.array([[price_per_unit, quantity, day, month]])
    features_poly = poly.transform(features)
    prediction = model.predict(features_poly)[0]
    st.success(f"Estimated Total Spent: ${prediction:.2f}")
