import streamlit as st
import pickle
import numpy as np


with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title(" Loan Approval Prediction")
st.write("Enter applicant details below")


gender_text = st.selectbox('Gender', ['Male', 'Female'])
Gender = 1 if gender_text == 'Male' else 0

married_text = st.selectbox('Married', ['Yes', 'No'])
Married = 1 if married_text == 'Yes' else 0

Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
Dependents = 3 if Dependents == '3+' else int(Dependents)

education_text = st.selectbox('Education', ['Graduate', 'Not Graduate'])
Education = 0 if education_text == 'Graduate' else 1

self_emp_text = st.selectbox('Self Employed', ['No', 'Yes'])
Self_Employed = 1 if self_emp_text == 'Yes' else 0

ApplicantIncome = st.number_input('Applicant Income', min_value=0)
CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
LoanAmount = st.number_input('Loan Amount', min_value=0)
Loan_Amount_Term = st.number_input('Loan Term (months)', value=360)

credit_text = st.selectbox('Credit History', ['Good (1)', 'Bad (0)'])
Credit_History = 1.0 if 'Good' in credit_text else 0.0

property_text = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
Property_Area = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}[property_text]

if st.button("Predict Loan Status"):

    input_data = np.array([[
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success(" Loan Approved")
    else:
        st.error(" Loan Rejected")
