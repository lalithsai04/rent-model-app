import streamlit as st
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load("price_model.pkl")
scaler = joblib.load("scaler.pkl")
le_loc = joblib.load("location_encoder.pkl")
le_bal = joblib.load("balcony_encoder.pkl")

st.title("🏠 House Rent Price Predictor")

# Inputs
area = st.number_input("Total Area (sqft)", min_value=100, max_value=20000, value=1000)
baths = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5, 6])
price_per_sqft = st.number_input("Price per Sqft", min_value=100.0, max_value=100000.0, value=5000.0)

location = st.selectbox("Location", le_loc.classes_)
balcony = st.selectbox("Balcony", le_bal.classes_)

if st.button("Predict Rent Price"):
    loc_encoded = le_loc.transform([location])[0]
    bal_encoded = le_bal.transform([balcony])[0]
    area_per_bath = area / (baths + 1)

    input_data = np.array([[area, baths, price_per_sqft, loc_encoded, bal_encoded, area_per_bath]])
    input_scaled = scaler.transform(input_data)

    price_pred = model.predict(input_scaled)[0]
    st.success(f"🏷️ Predicted Rent Price: ₹ {price_pred:,.0f}")