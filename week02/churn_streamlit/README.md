📉 Customer Churn Prediction App
Welcome to the Customer Churn Prediction App built with Streamlit and XGBoost! This interactive tool allows you to explore customer churn data and predict whether a customer is likely to churn based on key features.

📦 Features
🗃️ Dataset Overview: Displays the first few rows of your churn dataset.

📊 Churn Distribution: Visualizes the churn rate in your dataset.

🧠 Predict Churn: Use sliders and dropdowns to input customer features and get a real-time churn prediction powered by a pre-trained XGBoost model.

📋 How It Works
🔍 Dataset Overview
Loads a preprocessed CSV file (synthetic_churn_data_featured.csv)

Displays the first few rows for inspection.

📊 Churn Distribution
Shows a bar chart of customers who churned vs. who stayed.

🧠 Prediction Section
Set customer features using:

MonthlyCharges (slider)

SupportCalls (slider)

TenureMonths (slider)

ContractType (dropdown)

Feature engineering includes:

IsLongContract

SupportCallsPerMonth

Revenue

Model makes a prediction using a trained XGBoost classifier.

Outputs:

Prediction: churn or not churn

Probability score (e.g., 0.78)

📁 Model & Data Notes
🧠 The model is loaded from xgb_model.json, trained beforehand using customer features.

📊 Data is sourced from a synthetic churn dataset for demonstration and learning purposes.


✅ Requirements
Python 3.7+

Streamlit

XGBoost

Pandas

Install with:

bash
Copy
Edit
pip install streamlit xgboost pandas