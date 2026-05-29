
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('best_model.pkl')

st.title("Amazon Product Rating Predictor")

st.write("Enter product details below:")

price = st.number_input("Enter Product Price", min_value=0.0)
title_length = st.number_input("Enter Title Length", min_value=1)

if st.button("Predict Rating"):

    features = np.array([[price, title_length]])

    prediction = model.predict(features)

    st.success(f"Predicted Rating: {prediction[0]:.2f}")
