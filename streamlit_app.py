import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

st.set_page_config(page_title="Smart Lender", page_icon="🏦")

st.title("🏦 Smart Lender")
st.subheader("AI Loan Approval Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Status"):

    gender = label_encoders["Gender"].transform([gender])[0]
    married = label_encoders["Married"].transform([married])[0]
    dependents = label_encoders["Dependents"].transform([dependents])[0]
    education = label_encoders["Education"].transform([education])[0]
    self_employed = label_encoders["Self_Employed"].transform([self_employed])[0]
    property_area = label_encoders["Property_Area"].transform([property_area])[0]

    data = [[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]]

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("🎉 Loan Approved")
    else:
        st.error("❌ Loan Rejected")