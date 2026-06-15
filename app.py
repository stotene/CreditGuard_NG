import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="CreditGuard NG", page_icon=" ", layout="wide")

st.title("CreditGuard NG")
st.subheader("Intelligent Loan Approval Prediction System")
st.markdown("**Tech Crush | Group November**")
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

# Sidebar Inputs
st.sidebar.header("Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.sidebar.number_input("Applicant Income (₦)", min_value=0, value=5000)
co_applicant_income = st.sidebar.number_input("Co-applicant Income (₦)", min_value=0, value=0)
loan_amount = st.sidebar.number_input("Loan Amount (₦ '000)", min_value=10, value=100)
loan_amount_term = st.sidebar.number_input("Loan Term (months)", min_value=12, value=360)
credit_history = st.sidebar.selectbox("Credit History", ["Has History (1)", "No History (0)"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Preprocessing
def preprocess_input():
    data = {
        "gender": 1 if gender == "Male" else 0,
        "married": 1 if married == "Yes" else 0,
        "dependents": dependents,
        "education": 0 if education == "Graduate" else 1,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "applicant_income": applicant_income,
        "co_applicant_income": co_applicant_income,
        "loan_amount": loan_amount,
        "loan_amount_term": loan_amount_term,
        "credit_history": 1 if "Has" in credit_history else 0,
        "property_area": {"Rural":0, "Semiurban":1, "Urban":2}[property_area],
        "total_income": applicant_income + co_applicant_income,
        "income_loan_ratio": (applicant_income + co_applicant_income) / loan_amount if loan_amount > 0 else 0,
        "total_income_log": np.log1p(applicant_income + co_applicant_income),
        "loan_amount_log": np.log1p(loan_amount)
    }
    return pd.DataFrame([data])

input_df = preprocess_input()

if st.button("🔍 Predict Loan Approval", type="primary", use_container_width=True):
    prob = model.predict_proba(input_df)[0][1]
    risk_score = round((1 - prob) * 100, 1)
    
    if risk_score <= 30:
        status = "🟢 Low Risk - APPROVE"
    elif risk_score <= 60:
        status = "🟡 Medium Risk - REVIEW"
    else:
        status = "🔴 High Risk - REJECT"
    
    st.success(f"**Approval Probability: {prob:.1%}**")
    st.metric("Risk Score", f"{risk_score}/100", delta=status)
    
    # SHAP Explanation
    st.subheader("Decision Explanation")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    
    exp_df = pd.DataFrame({
        "Feature": [
            "Gender", "Married", "Dependents", "Education", "Self-Employed",
            "Applicant Income", "Co-applicant Income", "Loan Amount", "Loan Term",
            "Credit History", "Property Area", "Total Income", "Income-to-Loan Ratio",
            "Total Income Log", "Loan Amount Log"
        ],
        "Value": input_df.iloc[0].values,
        "SHAP Impact": sv.round(3)
    }).sort_values("SHAP Impact", key=abs, ascending=False)
    
    st.dataframe(exp_df, use_container_width=True)
