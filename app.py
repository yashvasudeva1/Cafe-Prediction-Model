import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Sidebar for model training (if data is available)
st.sidebar.header("Model Configuration")
degree = st.sidebar.slider("Polynomial Degree", min_value=1, max_value=6, value=4)
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2)

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
    
    # Create sample data for demonstration (you would replace this with your actual DataFrame)
    @st.cache_data
    def create_sample_data():
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data
        sample_data = []
        for _ in range(n_samples):
            item = np.random.choice(items)
            price = price_mapping.get(item, 2.0)  # Use .get() with default value
            qty = np.random.randint(1, 10)
            d = np.random.randint(1, 32)
            m = np.random.randint(1, 13)
            # Create some correlation for total spent
            total = price * qty + np.random.normal(0, 1)
            sample_data.append([item, price, qty, d, m, total])
        
        return pd.DataFrame(sample_data, columns=['Item', 'Price Per Unit', 'Quantity', 'Day', 'Month', 'Total Spent'])
    
    # Load or create data
    df = create_sample_data()
    
    # Train the model
    @st.cache_data
    def train_model(degree, test_size):
        x = df[['Price Per Unit', 'Quantity', 'Day', 'Month']]
        y = df['Total Spent']
        
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=42)
        
        pf = PolynomialFeatures(degree=degree)
        x_poly_train = pf.fit_transform(xtrain)
        x_poly_test = pf.transform(xtest)
        
        polynomial_model = LinearRegression()
        polynomial_model.fit(x_poly_train, ytrain)
        
        predicted = polynomial_model.predict(x_poly_test)
        mse_error = mean_squared_error(ytest, predicted)
        r2 = r2_score(ytest, predicted)
        
        return polynomial_model, pf, mse_error, r2, xtest, ytest, predicted
    
    # Train the model
    model, poly_features, mse, r2, xtest, ytest, predicted = train_model(degree, test_size)
    
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
    
    # Display model performance
    st.subheader("Model Performance")
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        st.metric(
            label="Mean Squared Error",
            value=f"{mse:.4f}"
        )
    
    with col2_2:
        st.metric(
            label="R² Score",
            value=f"{r2:.4f}"
        )



    
   
