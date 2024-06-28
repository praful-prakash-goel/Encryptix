# Importing Libraries
import pandas as pd
import pickle
import json
import streamlit as st

with open("./grad_booster.pkl", "rb") as f:
    model = pickle.load(f)

with open("./scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
st.title("Customer Churn Prediction")
st.write("Enter the following details to check if the customer is churned or not")

user_credit = st.slider("Enter the Credit Score", min_value = 350, max_value = 850)

user_loc = st.selectbox(
    "Select the Location",
    ('France', 'Germany', 'Spain')
)
if user_loc == 'France':
    user_loc = 0
elif user_loc == 'Germany':
    user_loc = 1
else:
    user_loc = 2

user_gender = st.selectbox(
    "Select Gender",
    ('Female', 'Male')
)
if user_gender == 'Female':
    user_gender = 0
else:
    user_gender = 1

user_age = st.text_input("Enter Age")

user_tenure = st.slider("Enter Tenure", min_value=0, max_value=10)

user_balance = st.text_input("Enter Balance")

user_prod = st.slider("Enter the number of products", min_value=1, max_value=4)

user_card = st.radio(
    "Does the user has Credit Card?",
    ["Yes", "No"]
)
if user_card == 'Yes':
    user_card = 1
else:
    user_card = 0

user_mem = st.radio(
    "Is the user an active member?",
    ["Yes", "No"]
)
if user_mem == 'Yes':
    user_mem = 1
else:
    user_mem = 0

user_salary = st.text_input("Enter the Estimated Salary of the user")

user_submit = st.button("Predict")

if user_submit:
    user_data = pd.DataFrame({
        'CreditScore': [user_credit],
        'Geography': [user_loc],
        'Gender': [user_gender],
        'Age': [user_age],
        'Tenure': [user_tenure],
        'Balance': [user_balance],
        'NumOfProducts': [user_prod],
        'HasCrCard': [user_card],
        'IsActiveMember': [user_mem],
        'EstimatedSalary': [user_salary]
    })

    user_data[['Balance', 'EstimatedSalary','CreditScore']] = scaler.transform(user_data[['Balance', 'EstimatedSalary','CreditScore']])
    prediction = model.predict(user_data)

    if prediction == 1:
        prediction_text = "The customer is churned"
    else:
        prediction_text = "The customer is not churned"

    st.subheader("Prediction Result:")
    st.write("Based on the provided information :", prediction_text)

user_report = st.button("Classification Report")
if user_report:
    st.image("./Classification_report.png", caption="Classification Report")