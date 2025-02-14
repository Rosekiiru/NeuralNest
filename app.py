import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')  # Make sure model is in the same folder as this script

# Streamlit UI setup
st.title("🏡 NeuralNest: Kenya Housing Price Prediction")
st.write("Enter the details of the property to predict its price.")

# Create input fields for user input
bedrooms = st.number_input("🛏 Bedrooms", min_value=0, value=3, step=1)
bathrooms = st.number_input("🛁 Bathrooms", min_value=0, value=2, step=1)
sq_mtrs = st.number_input("📏 Square Meters", min_value=1, value=150, step=1)
price_per_sq_meter = st.number_input("💰 Price per Sq Meter", min_value=1, value=8000, step=100)

# Predict button
if st.button("🔍 Predict Price"):
    # Prepare input data for prediction
    input_data = np.array([[bedrooms, bathrooms, sq_mtrs, price_per_sq_meter]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the predicted price
    st.success(f"🏠 **Estimated House Price:** KSh {prediction[0]:,.2f}")
