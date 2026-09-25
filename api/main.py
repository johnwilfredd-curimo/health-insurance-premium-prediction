from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.prediction import predict

app = FastAPI(
    title="Health Insurance Premium Prediction API",
    description=(
        "Predicts the annual health insurance premium to quote a customer. "
        "Segmented model: linear regression for ages 18-25, XGBoost for 26+."
    ),
    version="1.0.0",
)

class PremiumInput(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[35])
    number_of_dependants: int = Field(..., ge=0, le=20, examples=[2])
    income_lakhs: float = Field(..., ge=0, le=200, examples=[12])
    genetical_risk: int = Field(..., ge=0, le=5, examples=[2])
    insurance_plan: Literal["Bronze", "Silver", "Gold"]
    employment_status: Literal["Salaried", "Self-Employed", "Freelancer"]
    gender: Literal["Male", "Female"]
    marital_status: Literal["Unmarried", "Married"]
    bmi_category: Literal["Normal", "Obesity", "Overweight", "Underweight"]
    smoking_status: Literal["No Smoking", "Regular", "Occasional"]
    region: Literal["Northwest", "Southeast", "Northeast", "Southwest"]
    medical_history: Literal[
        "No Disease",
        "Diabetes",
        "High blood pressure",
        "Diabetes & High blood pressure",
        "Thyroid",
        "Heart Disease",
        "High blood pressure & Heart disease",
        "Diabetes & Thyroid",
        "Diabetes & Heart disease",
    ]

class PremiumOutput(BaseModel):
    predicted_premium: int
    currency: str = "INR"

@app.get("/ping")
def ping():
    return "Hello"

@app.post("/predict_premium", response_model=PremiumOutput)
def predict_premium(payload: PremiumInput):
    try:
        premium = predict(
            {
                "Age": payload.age,
                "Number of Dependants": payload.number_of_dependants,
                "Income in Lakhs": payload.income_lakhs,
                "Genetical Risk": payload.genetical_risk,
                "Insurance Plan": payload.insurance_plan,
                "Employment Status": payload.employment_status,
                "Gender": payload.gender,
                "Marital Status": payload.marital_status,
                "BMI Category": payload.bmi_category,
                "Smoking Status": payload.smoking_status,
                "Region": payload.region,
                "Medical History": payload.medical_history,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return PremiumOutput(predicted_premium=premium)