"""FastAPI model-serving layer for the loan-risk model."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train_model import predict_one, load_artifact


app = FastAPI(
    title="Loan Risk Scoring API",
    version="1.0.0",
    description="REST API for a portfolio loan-risk classification model.",
)


class LoanApplication(BaseModel):
    annual_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    debt_to_income: float = Field(..., ge=0, le=100)
    interest_rate: float = Field(..., ge=0, le=50)
    credit_utilization: float = Field(..., ge=0, le=200)
    open_accounts: int = Field(..., ge=0, le=100)
    delinquencies_2y: int = Field(..., ge=0, le=50)
    public_records: int = Field(..., ge=0, le=20)
    inquiries_6m: int = Field(..., ge=0, le=30)
    total_open_debt: float = Field(..., ge=0)
    employment_years: float = Field(..., ge=0, le=60)
    home_ownership: Literal["RENT", "MORTGAGE", "OWN"]
    verification_status: Literal["Verified", "Source Verified", "Not Verified"]
    loan_term: Literal["36 months", "60 months"]
    loan_purpose: Literal[
        "Debt consolidation",
        "Home improvement",
        "Major purchase",
        "Education",
        "Other",
    ]


@app.get("/health")
def health():
    artifact = load_artifact()
    return {
        "status": "healthy",
        "model_version": artifact["version"],
        "roc_auc": artifact["metrics"]["roc_auc"],
    }


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        return predict_one(application.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
