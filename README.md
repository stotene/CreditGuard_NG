# CreditGuard_NG

> **Intelligent Loan Approval Prediction System with Fairness & Risk Insights for Nigerian Microfinance**

[![Python](https://shields.io)](https://python.org)
[![Streamlit](https://shields.io)](https://streamlit.io)
[![Scikit-Learn](https://shields.io)](https://scikit-learn.org)
[![License: MIT](https://shields.io)](LICENSE)

An enterprise-grade, data-driven credit scoring solution engineered specifically to modernize risk assessment pipelines for Nigerian Microfinance Institutions (MFIs). This system balances predictive accuracy with algorithmic fairness to support thin-file applicants in underbanked ecosystems.

---

## 📌 Table of Contents

- [Business Case & Impact](#-business-case--impact)
- [Repository Architecture](#%EF%B8%8F-repository-architecture)
- [Tech Stack](#-tech-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Local Installation & Setup](#-local-installation--setup)
- [Streamlit Web Application](#-streamlit-web-application)
- [Contributors](#-contributors)
- [License](#-license)

---

## 💼 Business Case & Impact

Traditional credit scoring methodologies often exclude viable borrowers in emerging economies due to missing documentation, volatile informal income streams, or systemic demographic biases. 

`CreditGuard_NG` resolves these inefficiencies by deploying a machine learning engine that:
- **Reduces Non-Performing Loans (NPLs):** Optimizes classification thresholds to flag default vectors before capital allocation.
- **Drives Financial Inclusion:** Captures underlying relationships in non-traditional data markers to safely approve thin-file applications.
- **Ensures Unbiased Lending:** Evaluates demographic parity constraints to prevent structural bias across sensitive borrower profiles.

---

## 🗂️ Repository Architecture

The project maintains a clean separation of concerns between experimental development sandboxes and production-ready artifacts:

```text
├── Loan-Approval-Prediction.csv  # Raw, uncleaned source dataset
├── cleaned_dataset.csv           # Preprocessed training feature matrix
├── eda_cleaning.ipynb            # Jupyter Notebook: Data cleaning & EDA pipelines
├── model_training.ipynb          # Jupyter Notebook: Model exploration & tuning
├── best_model.joblib             # Serialized production-ready champion model
├── app.py                        # Reactive Streamlit web application layer
└── requirements.txt              # Pinpoint project dependency manifest
```

---

## 🛠️ Tech Stack

- **Languages:** `Python 3.8+`
- **Data Engineering:** `Pandas`, `NumPy`
- **Modeling & Evaluation:** `Scikit-Learn`, `Jupyter Notebook`
- **Artifact Serialization:** `Joblib`
- **Application Interface:** `Streamlit`

---

## 🔬 Machine Learning Pipeline

### 1. Data Engineering & EDA
- **Imputation Mapping:** Implements deterministic strategies to handle missing parameters without introducing bias.
- **Distribution Balancing:** Mitigates heavy-tailed distributions in informal income metrics using robust scaling and mathematical transforms.
- **Leakage Prevention:** Strictly separates pre-processing pipelines across validation folds to guarantee model generalization.

### 2. Model Evaluation
- Employs strict evaluation metrics (**ROC-AUC** and **F1-Score**) to optimize predictions against class imbalances.
- Compares structural tree-based classifiers and linear models to maximize operational stability.
- Serializes the optimal model directly into an inference-ready pipeline component (`best_model.joblib`).

---

## 🚀 Local Installation & Setup

To deploy the development workspace and model server locally, execute the following instructions in your terminal:

### Prerequisites
- Ensure Python 3.8 or higher is installed.
- Git version control configuration.

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd CreditGuard_NG
   ```

2. **Configure a virtual environment:**
   ```bash
   # Create environment
   python3 -m venv venv
   
   # Activate environment (macOS/Linux)
   source venv/bin/activate
   
   # Activate environment (Windows)
   venv\Scripts\activate
   ```

3. **Install exact project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the inference server application:**
   ```bash
   streamlit run app.py
   ```
   *The system will spin up an instance locally. Open your browser and navigate to `http://localhost:8501` to test the application.*

---

## 🖥️ Streamlit Web Application

The production layer maps inputs directly into the serialized inference asset via a user-friendly UI:
- **Interactive Forms:** Fields designed to seamlessly match field officer intake sheets.
- **Instant Probability Scoring:** Generates real-time approval status alongside risk confidence probabilities.
- **Decision Visibility:** Displays critical decision factors to give credit officers contextual insight into application failures or flags.

---

This project is licensed under the MIT License. See the `LICENSE` file for details.
