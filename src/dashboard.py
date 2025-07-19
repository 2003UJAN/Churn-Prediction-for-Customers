import streamlit as st
from src.preprocessing import load_and_preprocess
from src.model import train_models, evaluate
from src.explain import explain_model
import pandas as pd

def extract_key_metrics(report):
    """Extracts accuracy, precision, recall, f1 from classification report."""
    accuracy = report.get("accuracy", 0)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    return accuracy, precision, recall, f1

def show_metrics(title, accuracy, precision, recall, f1):
    st.markdown(f"### 🔍 {title}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy", f"{accuracy:.2%}")
    col2.metric("🧠 Precision", f"{precision:.2%}")
    col3.metric("📢 Recall", f"{recall:.2%}")
    col4.metric("📎 F1-Score", f"{f1:.2%}")

def app():
    st.title("Customer Churn Prediction Dashboard")

    with st.spinner("🔄 Loading and training models..."):
        X_train, X_test, y_train, y_test = load_and_preprocess()
        xgb, logreg = train_models(X_train, y_train)

    st.header("📊 Model Performance")

    # XGBoost
    xgb_report = evaluate(xgb, X_test, y_test)
    xgb_acc, xgb_prec, xgb_rec, xgb_f1 = extract_key_metrics(xgb_report)
    show_metrics("XGBoost Classifier", xgb_acc, xgb_prec, xgb_rec, xgb_f1)
    with st.expander("🔎 Full Report: XGBoost"):
        st.json(xgb_report)

    # Logistic Regression
    logreg_report = evaluate(logreg, X_test, y_test)
    logreg_acc, logreg_prec, logreg_rec, logreg_f1 = extract_key_metrics(logreg_report)
    show_metrics("Logistic Regression", logreg_acc, logreg_prec, logreg_rec, logreg_f1)
    with st.expander("🔎 Full Report: Logistic Regression"):
        st.json(logreg_report)

    st.header("📌 SHAP Explainability")
    explain_model(xgb, X_train)
    st.image("shap_summary.png", caption="SHAP Summary Plot (XGBoost)", use_column_width=True)
