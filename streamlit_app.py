"""Streamlit UI for the loan-risk model.

The UI can either:
1. Call the FastAPI endpoint when API_URL is configured, or
2. Run local inference from the same model artifact for easy demos.
"""

import os

import pandas as pd
import requests
import streamlit as st

from train_model import load_artifact, predict_one


st.set_page_config(
    page_title="Loan Risk Scoring",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Loan Risk Scoring")
st.caption("Portfolio demonstration: credit-risk classification with an API-ready ML pipeline.")

API_URL = os.getenv("API_URL", "").rstrip("/")

with st.sidebar:
    st.header("Applicant profile")

    annual_income = st.number_input("Annual income ($)", 18000.0, 600000.0, 85000.0, step=5000.0)
    loan_amount = st.number_input("Requested loan ($)", 2000.0, 100000.0, 25000.0, step=1000.0)
    debt_to_income = st.slider("Debt-to-income (%)", 2.0, 55.0, 18.0)
    interest_rate = st.slider("Interest rate (%)", 5.0, 30.0, 12.5)
    credit_utilization = st.slider("Credit utilization (%)", 1.0, 130.0, 45.0)
    open_accounts = st.slider("Open accounts", 1, 35, 8)
    delinquencies_2y = st.slider("Delinquencies in last 2 years", 0, 8, 0)
    public_records = st.slider("Public records", 0, 4, 0)
    inquiries_6m = st.slider("Credit inquiries (6 months)", 0, 8, 1)
    total_open_debt = st.number_input("Total open debt ($)", 1000.0, 500000.0, 55000.0, step=2500.0)
    employment_years = st.slider("Employment tenure (years)", 0.0, 35.0, 6.0)

    home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
    verification_status = st.selectbox(
        "Income verification",
        ["Verified", "Source Verified", "Not Verified"],
    )
    loan_term = st.selectbox("Loan term", ["36 months", "60 months"])
    loan_purpose = st.selectbox(
        "Loan purpose",
        ["Debt consolidation", "Home improvement", "Major purchase", "Education", "Other"],
    )

application = {
    "annual_income": annual_income,
    "loan_amount": loan_amount,
    "debt_to_income": debt_to_income,
    "interest_rate": interest_rate,
    "credit_utilization": credit_utilization,
    "open_accounts": open_accounts,
    "delinquencies_2y": delinquencies_2y,
    "public_records": public_records,
    "inquiries_6m": inquiries_6m,
    "total_open_debt": total_open_debt,
    "employment_years": employment_years,
    "home_ownership": home_ownership,
    "verification_status": verification_status,
    "loan_term": loan_term,
    "loan_purpose": loan_purpose,
}

if st.button("Assess loan risk", type="primary", use_container_width=True):
    with st.spinner("Scoring application..."):
        if API_URL:
            response = requests.post(f"{API_URL}/predict", json=application, timeout=20)
            response.raise_for_status()
            result = response.json()
            serving_mode = f"FastAPI: {API_URL}"
        else:
            result = predict_one(application)
            serving_mode = "Local model inference"

    probability = result["default_probability"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Default probability", f"{probability:.1%}")
    col2.metric("Risk band", result["risk_band"])
    col3.metric("Decision", result["decision"])

    if result["prediction"] == 1:
        st.error("Application flagged for higher-risk review.")
    else:
        st.success("Application falls into the lower-risk class.")

    st.caption(f"Serving mode: {serving_mode} | Model version: {result['model_version']}")

st.divider()

artifact = load_artifact()
metrics = artifact["metrics"]

m1, m2, m3 = st.columns(3)
m1.metric("Validation ROC-AUC", f"{metrics['roc_auc']:.3f}")
m2.metric("Validation accuracy", f"{metrics['accuracy']:.3f}")
m3.metric("Training rows", f"{metrics['training_rows']:,}")

st.subheader("Model context")
st.write(
    "The original notebook explored logistic regression, Random Forest, AdaBoost and "
    "Gradient Boosting, with class balancing via SMOTE. This deployment version packages "
    "the preprocessing, SMOTE step and Gradient Boosting classifier into one reproducible "
    "pipeline and exposes it through FastAPI and Streamlit."
)

st.info(
    "Portfolio note: this demo uses synthetic lending data. Do not represent the synthetic "
    "dataset or model as production data from a former employer."
)

# Small example table for a portfolio-friendly UI.
st.subheader("Current application payload")
st.dataframe(pd.DataFrame([application]), use_container_width=True)
