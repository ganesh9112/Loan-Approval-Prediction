from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and scaler
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

# Input schema
class LoanInput(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: int


@app.get("/")
def home():
    return {"message": "Loan Prediction API running 🚀"}


@app.post("/predict")
def predict(data: LoanInput):

    X = np.array([[
        data.Gender,
        data.Married,
        data.Dependents,
        data.Education,
        data.Self_Employed,
        data.ApplicantIncome,
        data.CoapplicantIncome,
        data.LoanAmount,
        data.Loan_Amount_Term,
        data.Credit_History,
        data.Property_Area
    ]])

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    return {
        "Loan_Status": "Approved" if prediction == 1 else "Rejected"
    }
