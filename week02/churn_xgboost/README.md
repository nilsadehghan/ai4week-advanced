📊 Customer Churn Prediction with XGBoost
This project uses a synthetic customer dataset to predict churn (whether a customer will leave). It features data preprocessing, feature engineering, handling class imbalance, and building an XGBoost model with evaluation metrics and visualizations.

📝 What This Code Does
Load Dataset
Loads synthetic_churn_data.csv with customer info.

Feature Engineering

Drops missing values

Creates new features like:

Frequent caller (calls > 2) 📞

Encodes ContractType 🔢

Calculates IsLongContract, Revenue, SupportCallsPerMonth 💰

Exploratory Data Analysis (EDA)
Plots distributions of key features by churn status using boxplots and countplots. 📈

Data Preparation

Removes unnecessary columns 🗑️

Handles class imbalance using SMOTE (synthetic minority oversampling) 🧪

Splits data into train/test sets (80/20) ✂️

Model Training

Uses XGBoost classifier with imbalance-aware parameters ⚖️

Trains on the balanced training set 🏋️‍♂️

Model Evaluation

Calculates precision, recall, accuracy, F1-score 🏆

Prints detailed classification report 📝

Plots ROC curve with AUC metric 🎯

Save Model
Saves the trained model to xgb_model.json 💾

🔧 Requirements
Python libraries:

pandas

matplotlib

seaborn

scikit-learn

xgboost

imblearn (for SMOTE)

numpy

Install them via pip if needed:

bash
Copy
Edit
pip install pandas matplotlib seaborn scikit-learn xgboost imblearn numpy
🚀 How to Run
Place your dataset synthetic_churn_data.csv in the same directory.

Run the script:

bash
Copy
Edit
python churn_prediction.py
View plots for feature distributions and ROC curve.

Check console output for model performance metrics.

Find the saved model file xgb_model.json for later use.

🧩 Notes
The dataset must include columns like:

SupportCalls, ContractType, TenureMonths, MonthlyCharges, Churn, etc.

Churn should be a binary target variable (0 = no churn, 1 = churn).

SMOTE handles class imbalance by oversampling minority class before training.

XGBoost's scale_pos_weight is set based on class distribution for better performance.

🎨 Visualizations Included
Monthly Charges vs Churn

Support Calls vs Churn

Contract Type vs Churn

Support Calls Per Month vs Churn

Revenue vs Churn

Contract Type count by churn

ROC Curve for model performance

