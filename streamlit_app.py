import streamlit as st
import gdown
import pickle
import numpy as np
import os

st.title("📈 Term Deposit Subscription Predictor")

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1m6N4vYimMUn4qkWnyLPFqdYlihqOZfrH"
    model_path = "model.pkl"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    with open(model_path, "rb") as file:
        return pickle.load(file)

model = load_model()

# Input form
st.subheader("Enter Client Information")

age = st.slider("Age", 18, 95)
duration = st.number_input("Duration of Last Call (in seconds)", min_value=0, value=100)
balance = st.number_input("Account Balance", min_value=0, value=1000)

# Predict
if st.button("Predict"):
    features = np.array([[age, balance, duration, 2, 3, 0, 1, 4, 2, 0, 1, 0, 0, 1, 0]])  # dummy values
    result = model.predict(features)
    st.success("✅ Likely to Subscribe" if result[0] == 1 else "❌ Not Likely to Subscribe")
