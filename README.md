# Loan Risk Prediction — Deployment-Ready Portfolio Project

This repository refactors the original `mu_sigma_AroraS.ipynb` loan-classification
notebook into a reproducible ML application.

## What changed

- Converted notebook-style analysis into reusable Python modules.
- Added a reproducible synthetic lending dataset generator.
- Added missing-value handling and categorical encoding.
- Kept the notebook's class-imbalance idea with SMOTE.
- Added a production-style scikit-learn/imblearn pipeline.
- Added FastAPI `/health` and `/predict` endpoints.
- Added a Streamlit interactive frontend.
- Added model serialization with Joblib.
- Added input validation with Pydantic.
- Made the app deployable from GitHub using `requirements.txt`.

## Important portfolio / resume note

The original notebook is a useful foundation, but the synthetic data in this version
is not historical employer data. Describe this as a portfolio project or as an
independent reconstruction inspired by lending-risk workflows. Do not claim that
you built this exact application at a former employer unless that is factually true.

## Files

- `train_model.py` — data generation, preprocessing, SMOTE, model training and inference.
- `api.py` — FastAPI REST API.
- `streamlit_app.py` — Streamlit frontend.
- `requirements.txt` — dependencies.
- `artifacts/` — generated model artifact after training.

## Run locally

### 1. Create environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This creates `artifacts/loan_risk_model.joblib`.

### 4. Start FastAPI

```bash
uvicorn api:app --reload
```

Open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

### 5. Start Streamlit

In a second terminal:

```bash
streamlit run streamlit_app.py
```

By default Streamlit performs local inference.

To make Streamlit call the FastAPI service instead:

Windows PowerShell:
```powershell
$env:API_URL="http://127.0.0.1:8000"
streamlit run streamlit_app.py
```

macOS/Linux:
```bash
export API_URL=http://127.0.0.1:8000
streamlit run streamlit_app.py
```

## Deployment architecture

```text
User
  |
  v
Streamlit UI
  |
  | HTTP POST /predict
  v
FastAPI
  |
  v
Serialized ML Pipeline
  |
  +--> preprocessing
  +--> SMOTE (training only)
  +--> Gradient Boosting
  |
  v
Risk probability + decision
```

The architecture follows the deployment themes covered by the Udemy bootcamp:
turning notebook code into Python modules, exposing a model through FastAPI,
building a Streamlit UI, and preparing the project for cloud deployment.

## Resume wording

A truthful portfolio-style bullet could be:

**Loan Risk Prediction & ML Deployment | Python, Scikit-learn, FastAPI, Streamlit**
- Built an end-to-end loan-risk classification pipeline with preprocessing,
  categorical encoding, SMOTE-based class balancing and Gradient Boosting.
- Exposed real-time scoring through a FastAPI REST endpoint with Pydantic
  validation and served an interactive Streamlit application for scenario testing.
- Added reproducible model serialization, health checks and deployment-ready
  dependency management using GitHub.

Do not state an employer name for this project unless the work was actually
performed there.
