
import streamlit as st
import pandas as pd
import joblib

def load_model():
    retun joblib.load("churn_prediction_model_v1_0.joblib")

model = load_model

st.title("Customer Churn Prediction App")
st.write("The Customer Churn Prediction App is an internal tool for bank staff that predicts whether customers are at risk of churning based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")

CreditScore = st.number_input("Credit Score (customer's credit score)", min_value=300, max_value=900, value=650)
Geography = st.selectbox("Geography (country where the customer resides)", ["France", "Germany", "Spain"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (number of years the customer has been with the bank)", value=12)
Balance = st.number_input("Account Balance (customer’s account balance)", min_value=0.0, value=10000.0)
NumberOfProducts = st.number_input("Number of Products (number of products the customer has with the bank)", min_value=1, value=1)
HasCrCard = st.selectbox("Has credit card (customer has credit card)", ["Yes", "No"])
IsActiveMember = st.selectbox("Is active member(Customer is active member?", ["Yes", "No"])
EstimatedSalary = st.number_input("Estimated annual salary (Customer estimated salary", min_value = 0.0, value = 50000.0)

input_data = pd.DataFrame([{
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumberOfProducts': NumberOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}])

classification_threshold = 0.45

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0,1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Churn" if prediction == 1 else "Not Churn"
    print(f"Based on the customer's provided information, the customer will {result}.")
