"""
Loan Risk Prediction - improved training and inference pipeline.

This is the deployment-oriented replacement for the earlier train_model.py.

The project uses a synthetic lending dataset for portfolio demonstration.
It should NOT be presented as historical borrower data from an employer.

Run:
    python train_model.py

The script:
- generates reproducible synthetic lending data
- introduces realistic missing values
- preprocesses numeric/categorical features
- handles class imbalance with SMOTE inside the training pipeline
- compares Logistic Regression, Random Forest and Gradient Boosting
- selects the best model using validation ROC-AUC
- saves the trained pipeline with Joblib
- saves evaluation metrics to artifacts/model_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "loan_risk_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"

NUMERIC_FEATURES = [
    "annual_income",
    "loan_amount",
    "debt_to_income",
    "interest_rate",
    "credit_utilization",
    "open_accounts",
    "delinquencies_2y",
    "public_records",
    "inquiries_6m",
    "total_open_debt",
    "employment_years",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "loan_term",
    "loan_purpose",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def generate_synthetic_lending_data(
    n_samples: int = 15000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a reproducible synthetic lending dataset."""

    rng = np.random.default_rng(seed)

    annual_income = np.clip(
        rng.lognormal(mean=11.0, sigma=0.55, size=n_samples),
        18_000,
        600_000,
    )

    loan_amount = np.clip(
        rng.lognormal(mean=9.8, sigma=0.60, size=n_samples),
        2_000,
        100_000,
    )

    debt_to_income = np.clip(rng.normal(19, 7, n_samples), 2, 55)
    interest_rate = np.clip(rng.normal(13.5, 3.5, n_samples), 5, 30)
    credit_utilization = np.clip(rng.normal(55, 20, n_samples), 1, 130)

    open_accounts = np.clip(rng.poisson(9, n_samples), 1, 35)
    delinquencies_2y = np.clip(rng.poisson(0.50, n_samples), 0, 8)
    public_records = np.clip(rng.poisson(0.08, n_samples), 0, 4)
    inquiries_6m = np.clip(rng.poisson(1.3, n_samples), 0, 8)

    total_open_debt = np.clip(
        annual_income * rng.uniform(0.15, 1.10, n_samples),
        1_000,
        500_000,
    )

    employment_years = np.clip(rng.normal(7, 4.5, n_samples), 0, 35)

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"],
        n_samples,
        p=[0.42, 0.48, 0.10],
    )

    verification_status = rng.choice(
        ["Verified", "Source Verified", "Not Verified"],
        n_samples,
        p=[0.45, 0.38, 0.17],
    )

    loan_term = rng.choice(
        ["36 months", "60 months"],
        n_samples,
        p=[0.72, 0.28],
    )

    loan_purpose = rng.choice(
        [
            "Debt consolidation",
            "Home improvement",
            "Major purchase",
            "Education",
            "Other",
        ],
        n_samples,
        p=[0.48, 0.18, 0.13, 0.08, 0.13],
    )

    # Synthetic business logic:
    # higher leverage, utilization, delinquencies and inquiries increase risk;
    # higher income and longer employment reduce risk.
    #
    # This is intentionally synthetic. It is not based on employer/customer data.
    risk_signal = (
        1.5
        * (
            0.085 * (debt_to_income - 15)
            + 0.045 * (interest_rate - 10)
            + 0.028 * (credit_utilization - 45)
            + 0.55 * delinquencies_2y
            + 0.70 * public_records
            + 0.16 * inquiries_6m
            + 0.000008 * loan_amount
            - 0.000006 * annual_income
            - 0.04 * employment_years
            + 0.50 * (loan_term == "60 months")
            + 0.25 * (home_ownership == "RENT")
            + 0.30 * (verification_status == "Not Verified")
            + 0.18 * (loan_purpose == "Other")
        )
        - 3.5
    )

    probability = 1 / (1 + np.exp(-risk_signal))
    default_flag = rng.binomial(1, probability)

    data = pd.DataFrame(
        {
            "annual_income": annual_income.round(2),
            "loan_amount": loan_amount.round(2),
            "debt_to_income": debt_to_income.round(2),
            "interest_rate": interest_rate.round(2),
            "credit_utilization": credit_utilization.round(2),
            "open_accounts": open_accounts.astype(int),
            "delinquencies_2y": delinquencies_2y.astype(int),
            "public_records": public_records.astype(int),
            "inquiries_6m": inquiries_6m.astype(int),
            "total_open_debt": total_open_debt.round(2),
            "employment_years": employment_years.round(1),
            "home_ownership": home_ownership,
            "verification_status": verification_status,
            "loan_term": loan_term,
            "loan_purpose": loan_purpose,
            "default_flag": default_flag,
        }
    )

    # Introduce a small amount of missing data so the deployment pipeline
    # demonstrates production-style imputation.
    for column in [
        "annual_income",
        "debt_to_income",
        "credit_utilization",
        "employment_years",
    ]:
        indices = rng.choice(
            n_samples,
            size=max(1, int(n_samples * 0.03)),
            replace=False,
        )
        data.loc[indices, column] = np.nan

    return data


def make_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer."""

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(classifier) -> ImbPipeline:
    """Build a complete preprocessing -> SMOTE -> model pipeline."""

    return ImbPipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("smote", SMOTE(sampling_strategy=0.85, random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )


def get_candidate_models() -> dict:
    """Return candidate models for comparison."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            C=0.7,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=4,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a classifier using probability-based and threshold metrics."""

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_save_model(n_samples: int = 15000) -> dict:
    """Train candidates, select by ROC-AUC, and save the winning pipeline."""

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_lending_data(n_samples=n_samples)

    X = df[FEATURES]
    y = df["default_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results = {}
    trained_models = {}

    print("=" * 70)
    print("LOAN RISK MODEL TRAINING")
    print("=" * 70)
    print(f"Dataset size: {len(df):,}")
    print(f"Default rate: {y.mean():.1%}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print()

    for name, classifier in get_candidate_models().items():
        print(f"Training {name}...")

        model = build_pipeline(classifier)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        results[name] = metrics
        trained_models[name] = model

        print(
            f"  Accuracy={metrics['accuracy']:.3f} | "
            f"Precision={metrics['precision']:.3f} | "
            f"Recall={metrics['recall']:.3f} | "
            f"F1={metrics['f1']:.3f} | "
            f"ROC-AUC={metrics['roc_auc']:.3f}"
        )

    best_model_name = max(
        results,
        key=lambda model_name: results[model_name]["roc_auc"],
    )

    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    artifact = {
        "model": best_model,
        "features": FEATURES,
        "threshold": 0.50,
        "model_name": best_model_name,
        "metrics": best_metrics,
        "all_model_results": results,
        "version": "2.0.0",
        "data_note": "Synthetic lending data generated for portfolio demonstration.",
    }

    joblib.dump(artifact, MODEL_PATH)

    # Save JSON without the model objects.
    json_metrics = {
        "selected_model": best_model_name,
        "selected_model_metrics": best_metrics,
        "all_model_metrics": results,
        "dataset": {
            "rows": int(len(df)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "default_rate": float(y.mean()),
        },
        "version": "2.0.0",
    }

    METRICS_PATH.write_text(
        json.dumps(json_metrics, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)
    print(f"Selected model: {best_model_name}")
    print(f"Accuracy:       {best_metrics['accuracy']:.3f}")
    print(f"Precision:      {best_metrics['precision']:.3f}")
    print(f"Recall:         {best_metrics['recall']:.3f}")
    print(f"F1 Score:       {best_metrics['f1']:.3f}")
    print(f"ROC-AUC:        {best_metrics['roc_auc']:.3f}")
    print()
    print(f"Model saved to:   {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    return json_metrics


def load_artifact():
    """Load the trained model, training it first if necessary."""

    if not MODEL_PATH.exists():
        train_and_save_model()

    return joblib.load(MODEL_PATH)


def predict_one(application: dict) -> dict:
    """Score a single loan application."""

    artifact = load_artifact()

    model = artifact["model"]
    threshold = artifact["threshold"]

    row = pd.DataFrame([application])

    probability = float(model.predict_proba(row)[:, 1][0])
    prediction = int(probability >= threshold)

    if probability < 0.30:
        risk_band = "Low"
    elif probability < 0.60:
        risk_band = "Medium"
    else:
        risk_band = "High"

    return {
        "default_probability": round(probability, 4),
        "prediction": prediction,
        "decision": (
            "Higher-risk / Review"
            if prediction
            else "Lower-risk / Eligible"
        ),
        "risk_band": risk_band,
        "model_name": artifact["model_name"],
        "model_version": artifact["version"],
    }


if __name__ == "__main__":
    train_and_save_model()
