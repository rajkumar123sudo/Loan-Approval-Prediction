import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("loan_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0,1,2,3])
education = st.selectbox("Education", ["Graduate","Not Graduate"])
self_emp = st.selectbox("Self Employed", ["No","Yes"])

income = st.number_input("Applicant Income", min_value=0)

co_income = st.number_input("Coapplicant Income", min_value=0)

loan_amount = st.number_input("Loan Amount", min_value=0)

loan_term = st.number_input("Loan Amount Term", value=360)

credit = st.selectbox("Credit History", [0,1])

property_area = st.selectbox(
    "Property Area",
    ["Rural","Semiurban","Urban"]
)

if st.button("Predict"):

    gender = 1 if gender=="Male" else 0
    married = 1 if married=="Yes" else 0
    education = 1 if education=="Graduate" else 0
    self_emp = 1 if self_emp=="Yes" else 0

    semiurban = 1 if property_area=="Semiurban" else 0
    urban = 1 if property_area=="Urban" else 0

    data = np.array([[gender, married, dependents, education,
                      self_emp, income, co_income,
                      loan_amount, loan_term,
                      credit, semiurban, urban]])

    data = scaler.transform(data)

    pred = model.predict(data)

    if pred[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")    