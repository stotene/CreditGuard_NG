CreditGuard_NG🏦 
Intelligent Loan Approval Prediction System with Fairness & Risk Insights for Nigerian Microfinance.

CreditGuard_NG is a machine learning-powered decision support system designed to help Nigerian microfinance institutions assess loan applications. By analyzing applicant data, the system predicts loan approval viability while actively mitigating bias and providing transparent risk insights.
🚀 Key Features

Loan Approval Prediction: High-accuracy classification of loan defaults and approvals.
Risk Insights: Detailed feature analysis to pinpoint high-risk application metrics.
Fairness Metrics: Designed to address demographic parity and ensure unbiased lending decisions.
Interactive Web App: Simple user interface built with Streamlit for real-time loan evaluation.
Reproducible Pipeline: Complete end-to-end data cleaning, EDA, and model training notebooks.

📁 Repository Structuretext├── Loan-Approval-Prediction.csv  # Original, raw dataset
├── cleaned_dataset.csv           # Preprocessed dataset used for training
├── eda_cleaning.ipynb            # Jupyter Notebook for Data Cleansing & EDA
├── model_training.ipynb          # Jupyter Notebook for Model Training & Evaluation
├── best_model.joblib             # Serialized production-ready ML model
├── app.py                        # Streamlit web application script
├── requirements.txt              # Project dependencies and libraries
└── README.md                     # Project documentation

🛠️ Technology
StackLanguage: Python 3.x
Data Analysis: Pandas, NumPy, Jupyter Notebook
Machine Learning: Scikit-Learn, JoblibWeb 
Framework: Streamlit⚙️ Installation & Setup

Follow these steps to run the project locally on your machine.

1. Clone the Repositorybashgit clone https://github.com
cd CreditGuard_NG
2. Create a Virtual Environmentbash# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependenciesbashpip install -r requirements.txt
4. Run the Streamlit Applicationbashstreamlit run app.py
The app should automatically open in your browser at http://localhost:8501.
📊 Machine Learning Workflow

Data Cleaning (eda_cleaning.ipynb): Handles missing data, outlines outliers, and performs initial feature engineering on Nigerian demographic profiles.
Model Exploration (model_training.ipynb): Trains and compares various classification algorithms to select the optimal model.
Deployment (best_model.joblib): Exports the highest-performing model for seamless API or UI integration.
