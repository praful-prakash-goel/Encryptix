# Importing Libraries
import pandas as pd
import pickle
import json
import streamlit as st

with open('C:/Users/prafu/Python/Projects/Credit Card Fraud Detection/metadata.json', 'r') as f:
    meta = json.load(f)

st.title("Credit Card Fraud Detection")
st.write("Enter the following transaction details to check whether the transaction is legitimate or fraud.")

user_time = st.selectbox(
    "Select the time of the day",
    ('Morning', 'Afternoon', 'Evening', 'Night')
)
if user_time == 'Morning':
    user_time = 0
elif user_time == 'Afternoon':
    user_time = 1
elif user_time == 'Evening':
    user_time = 2
else:
    user_time = 3


user_category = st.selectbox(
    "Enter category",
    tuple(meta['category'])
)

user_age = st.text_input("Enter your age", key = "age")
user_amt = st.text_input("Enter the amount", key = "amount")

user_gender = st.selectbox(
    "Select your gender",
    ('Male', 'Female')
)
if user_gender == 'Male':
    user_gender = 1
else:
    user_gender = 0

user_job = st.selectbox(
    "Select your job",
    tuple(meta['job'])          
)

user_submit = st.button("Predict")

with open('C:/Users/prafu/Python/Projects/Credit Card Fraud Detection/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('C:/Users/prafu/Python/Projects/Credit Card Fraud Detection/encoder_model.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

if user_submit:
    user_data = pd.DataFrame({
        'time': [user_time],
        'category': [user_category],
        'age': [user_age],
        'amt': [user_amt],
        'gender': [user_gender],
        'job':[user_job]
    })

    user_data = target_encoder.transform(user_data)
    
    prediction = model.predict(user_data)
    
    if(prediction == 1):
        prediction_text = 'The transaction is fraud'
    else:
        prediction_text = 'The transaction is legitimate'
    
    st.subheader("Prediction Result:")
    st.write("Based on the provided information :", prediction_text)

user_report = st.button("Classification Report")
if user_report:
    st.image("C:/Users/prafu/Python/Projects/Credit Card Fraud Detection/Classification_report.jpg", caption="Classification Report")