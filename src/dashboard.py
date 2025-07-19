import streamlit as st
from src.preprocessing import load_and_preprocess
from src.model import train_models, evaluate
from src.explain import explain_model

def app():
    st.title("Customer Churn Prediction")

    with st.spinner("Loading and training models..."):
        X_train, X_test, y_train, y_test = load_and_preprocess()
        xgb, logreg = train_models(X_train, y_train)

    st.header("📊 Model Performance")

    st.subheader("XGBoost Classifier")
    xgb_report = evaluate(xgb, X_test, y_test)
    st.json(xgb_report)

    st.subheader("Logistic Regression")
    logreg_report = evaluate(logreg, X_test, y_test)
    st.json(logreg_report)

    st.header("🔍 SHAP Explainability for XGBoost")
    explain_model(xgb, X_train)
    st.image("shap_summary.png", use_column_width=True)
