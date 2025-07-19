import streamlit as st
import pandas as pd
from src.preprocessing import load_and_preprocess
from src.model import train_models, evaluate
from src.explain import explain_model
import matplotlib.pyplot as plt

def app():
    st.title("Blinkit Churn Prediction Dashboard")

    X_train, X_test, y_train, y_test = load_and_preprocess()
    xgb, logreg = train_models(X_train, y_train)

    st.subheader("Model Evaluation")
    st.write("**XGBoost:**")
    st.json(evaluate(xgb, X_test, y_test))

    st.write("**Logistic Regression:**")
    st.json(evaluate(logreg, X_test, y_test))

    st.subheader("SHAP Explainability (XGBoost)")
    explain_model(xgb, X_train)
    st.image("shap_summary.png", caption="SHAP Summary Plot", use_column_width=True)

