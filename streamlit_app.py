import streamlit as st
import gdown
import pickle
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Term Deposit Predictor", layout="centered")
st.title("📈 Term Deposit Subscription Predictor")

# -- Download and cache the model
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1m6N4vYimMUn4qkWnyLPFqdYlihqOZfrH"
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# -- Input form
st.subheader("Enter Client Information")

age = st.slider("Age", 18, 95, 30)
balance = st.number_input("Account Balance", min_value=0, value=1000)
duration = st.number_input("Duration of Last Call (in seconds)", min_value=0, value=100)

# Dummy selections to match model input shape
# You can replace these with dropdowns later
job = 2
education = 3
default = 0
housing = 1
loan = 0
contact = 1
month = 4
day_of_week = 2
campaign = 1
previous = 0
poutcome = 0
emp_var_rate = -1.8
cons_price_idx = 92.9

# Button outside of any if-block to guarantee visibility
if st.button("Predict"):
    feature_values = [
        age,
        balance,
        duration,
        job,
        education,
        default,
        housing,
        loan,
        contact,
        month,
        day_of_week,
        campaign,
        previous,
        poutcome,
        emp_var_rate,
        cons_price_idx
    ]
    
    cols = [
        'age', 'balance', 'duration', 'job', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'campaign', 'previous', 'poutcome',
        'emp_var_rate', 'cons_price_idx'
    ]

    input_df = pd.DataFrame([feature_values], columns=cols)

    result = model.predict(input_df.values)

    st.success("Likely to Subscribe" if result[0] == 1 else "Not Likely to Subscribe")
