# Loan Risk Scoring & Default Prediction

An end-to-end machine learning project for predicting loan default risk using applicant financial, credit-history, employment, and demographic information.

The project covers the complete ML workflow — from data preprocessing and model development to REST API integration and an interactive Streamlit application.

## Project Overview

Credit risk assessment is an important use case in financial services, where accurately identifying potentially high-risk loan applications can support better lending decisions.

This project builds a binary classification model to estimate the likelihood of loan default and exposes the trained model through both:

- A **FastAPI REST API** for programmatic predictions
- A **Streamlit web application** for interactive predictions

## Key Features

- Data cleaning and preprocessing
- Missing-value analysis
- Categorical feature encoding
- Feature engineering
- Train/test data splitting
- Classification model development
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Trained model saved using Joblib
- FastAPI prediction endpoint
- Interactive Streamlit interface
- Production-oriented project structure

## Model Performance

The current Logistic Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 0.772 |
| Precision | 0.478 |
| Recall | 0.695 |
| F1 Score | 0.566 |
| ROC-AUC | 0.827 |

The ROC-AUC score indicates that the model provides useful separation between higher- and lower-risk applications. Precision and recall can be further optimized depending on the business objective and the cost of false positives versus false negatives.

## Technology Stack

**Programming & Data**
- Python
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- Logistic Regression
- Classification metrics
- Joblib

**API**
- FastAPI
- Uvicorn

**Application**
- Streamlit

**Development**
- Git
- GitHub
- VS Code

## Project Architecture

User

Streamlit Web App

FastAPI REST API

Trained ML Model

Loan Risk Prediction
