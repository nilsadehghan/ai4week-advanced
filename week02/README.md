📉 Week 02 Challenge - Customer Churn Prediction
🔍 Overview
This project aims to build a customer churn prediction system using a synthetic dataset. It involves two main parts:

Model training and evaluation (Python script)

Interactive prediction app using Streamlit

The project walks through the entire machine learning pipeline from data preprocessing and feature engineering to model training, evaluation, and deployment via a user-friendly web interface.

🛠️ 1. Model Training and Evaluation
✨ Key Features
Performs feature engineering:

Encodes categorical features

Creates derived features such as Revenue, SupportCallsPerMonth, and IsLongContract

Handles class imbalance using SMOTE

Splits data into training and testing sets

Trains an XGBoost classifier with scale_pos_weight to address imbalance

Evaluates the model with:

Accuracy

F1 Score

Precision & Recall

Classification Report

ROC Curve & AUC

Saves the trained model to a file (xgb_model.json)

📊 Visualizations Included
Boxplots for feature distribution by churn

Count plots by contract type and churn

ROC Curve

🌐 2. Streamlit App 
🧠 Functionality
An interactive web app to:

Preview the dataset

Visualize churn distribution

Input customer features via sliders and dropdowns

Get churn prediction and probability using the pre-trained model

📌 Input Features
MonthlyCharges: Slider (0–100)

SupportCalls: Slider (0–200)

TenureMonths: Slider (0–100)

ContractType: Dropdown (Monthly, 1-Year, 2-Year)

The app computes engineered features (Revenue, SupportCallsPerMonth, etc.) from these inputs and feeds them into the model for prediction.

📌 Notes
The dataset used is synthetic, but mimics real-world customer churn scenarios.

scale_pos_weight and SMOTE are crucial for improving performance on imbalanced classes.

This project can be easily extended to support real datasets or additional features.
