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

import pandas as pd

cols = [
    'age', 'balance', 'duration', 'job', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign',
    'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx'
]

values = [age, balance, duration, 2, 3, 0, 1, 0, 1, 4, 2, 0, 1, 0, -1.8, 92.893, -46.2]

# Create DataFrame
input_df = pd.DataFrame([values], columns=cols)

# Predict
st.write("Model expects:", model.n_features_in_, "features")
result = model.predict(input_df.values)
