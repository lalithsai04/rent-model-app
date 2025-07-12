# Rent Price Prediction App

This project uses a Ridge Regression model to predict house rent prices based on inputs like total area, number of bathrooms, price per square foot, location, and balcony availability.

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the Streamlit app:

```
streamlit run app.py
```

## Files
- `app.py`: Streamlit app interface
- `price_model.pkl`: Trained ML model
- `scaler.pkl`: Feature scaler
- `location_encoder.pkl`, `balcony_encoder.pkl`: Encoders for categorical data