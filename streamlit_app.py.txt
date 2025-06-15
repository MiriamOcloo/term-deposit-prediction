import streamlit as st
import gdown
import pickle
import numpy as np

st.title("📈 Term Deposit Subscription Predictor")

# Download model from Google Drive (replace with your real ID)
url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
gdown.download(url, "model.pkl", quiet=False)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# User inputs
age = st.slider("Age", 18, 95)
duration = st.number_input("Duration of Last Call (in seconds)", value=100)
balance = st.number_input("Account Balance", value=1000)

# Add more fields as needed...

# Predict
if st.button("Predict"):
    features = np.array([[age, balance, duration, 2, 3, 0, 1, 4, 2, 0, 1, 0, 0, 1, 0]])  # example inputs
    result = model.predict(features)
    st.success("Likely to Subscribe" if result[0] == 1 else "Not Likely to Subscribe")
