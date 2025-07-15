import streamlit as st
import numpy as np
import joblib

# ── Load model + preprocessors ─────────────────────────────────────
model  = joblib.load("price_model.pkl")
scaler = joblib.load("scaler.pkl")
le_bal = joblib.load("balcony_encoder.pkl")   # only balcony encoder needed

st.title("🏠 House Price Predictor")

# ── User inputs ───────────────────────────────────────────────────
area            = st.number_input("Total Area (sqft)", 100, 20000, 1000)
baths           = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5, 6])
price_per_sqft  = st.number_input("Price per Sqft", 100.0, 100000.0, 5000.0)
balcony         = st.selectbox("Balcony", le_bal.classes_)

# ── Prediction ────────────────────────────────────────────────────
if st.button("Predict Rent Price"):
    bal_encoded   = le_bal.transform([balcony])[0]
    area_per_bath = area / (baths + 1)

    features      = np.array([[area, baths, price_per_sqft,
                               0, bal_encoded, area_per_bath]])  # 0 in place of location
    features_std  = scaler.transform(features)
    price_pred    = model.predict(features_std)[0]

    st.success(f"🏷️ Predicted Rent Price: ₹ {price_pred:,.0f}")
