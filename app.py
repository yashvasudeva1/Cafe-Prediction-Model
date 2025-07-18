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
prices = [2.0, 3.0, 1.0, 5.0, 4.0, 1.5, 0.0, 2.5]  # Added price for Juice

# Create price mapping dictionary
price_mapping = dict(zip(items, prices))

# Sidebar for model config info (can be used later for notes)
st.sidebar.header("Model Info")
st.sidebar.info("Using a pre-trained polynomial regression model.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Parameters")

    # Item selection dropdown
    selected_item = st.selectbox(
        "Select Item",
        options=items,
        index=0,
        help="Choose an item from the dropdown"
    )

    # Auto-populate price based on selection
    price_per_unit = price_mapping[selected_item]
    st.number_input(
        "Price Per Unit",
        value=price_per_unit,
        disabled=True,
        help=f"Auto-populated based on selected item: {selected_item}"
    )

    # Other input parameters
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=100,
        value=1,
        help="Enter the quantity"
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1,
        help="Enter the day (1-31)"
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1,
        help="Enter the month (1-12)"
    )

with col2:
    st.header("Prediction Results")

    @st.cache_resource
    def load_model_and_features():
        model = joblib.load("polynomial_model.pkl")
        poly_features = joblib.load("poly_features.pkl")
        return model, poly_features

    try:
        model, poly_features = load_model_and_features()

        # Make prediction for user input
        user_input = np.array([[price_per_unit, quantity, day, month]])
        user_input_poly = poly_features.transform(user_input)
        prediction = model.predict(user_input_poly)[0]

        # Display prediction
        st.metric(
            label="Predicted Profit",
            value=f"${prediction:.2f}",
            delta=f"Item: {selected_item}"
        )

        # Note: Performance metrics not available unless saved separately
        st.subheader("Model Performance")
        st.info("Model performance metrics (MSE, R²) are not available in this version.")

    except Exception as e:
        st.error(f"Error loading model or making prediction: {e}")

