import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(page_title="Polynomial Regression Predictor", layout="wide")

# Title and description
st.title("Polynomial Regression Predictor")
st.markdown("Predict total spending based on item selection and other factors")

# Item and price mapping
items = ['Coffee', 'Cake', 'Cookie', 'Salad', 'Smoothie', 'Sandwich', 'Tea', 'Juice']
prices = [2.0, 3.0, 1.0, 5.0, 4.0, 1.5, 0.0, 2.5]

# Create price mapping dictionary
price_mapping = dict(zip(items, prices))

# Sidebar (Optional Info)
st.sidebar.header("Model Info")
st.sidebar.info("Model and transformer loaded from uploaded `.pkl` files.")

# Input and Output Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Parameters")

    # Item selection
    selected_item = st.selectbox("Select Item", options=items)
    price_per_unit = price_mapping[selected_item]

    st.number_input("Price Per Unit", value=price_per_unit, disabled=True)
    quantity = st.number_input("Quantity", min_value=1, max_value=100, value=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)

with col2:
    st.header("Prediction Results")

    @st.cache_resource
    def load_model_and_transformer():
        model = joblib.load("/mnt/data/polynomial_model.pkl")
        transformer = joblib.load("/mnt/data/poly_features.pkl")
        return model, transformer

    try:
        model, poly_features = load_model_and_transformer()

        # Prepare input and make prediction
        user_input = np.array([[price_per_unit, quantity, day, month]])
        input_poly = poly_features.transform(user_input)
        prediction = model.predict(input_poly)[0]

        st.metric(label="Predicted Spending", value=f"${prediction:.2f}", delta=f"Item: {selected_item}")

        st.subheader("Model Performance")
        st.info("Pretrained model loaded — performance metrics are not shown in this demo.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

