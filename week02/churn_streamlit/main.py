import streamlit as st
import xgboost as xgb
import pandas as pd

# Set page configuration: title and layout centered
st.set_page_config(page_title="📉customer churn", layout="centered")


st.title("📉customer churn")

# Cache the data loading function so it only runs once unless the data changes
@st.cache_data
def load_data():
    # Load the dataset from a CSV file
    return pd.read_csv("ai_data_advanced/week02/synthetic_churn_data_featured.csv")

# Load the dataset into a dataframe
df = load_data()

# Load pre-trained XGBoost model from a JSON file
model = xgb.XGBClassifier()
model.load_model("ai_data_advanced/week02/xgb_model.json")

# Show a subheader and display the first few rows of the dataset for overview
st.subheader("🔍Dataset Overview")
st.write(df.head())

# Show a bar chart of the churn distribution in the dataset
st.subheader("📊Churn Distribution")
st.bar_chart(df["Churn"].value_counts())

# User input section to predict churn based on features
st.subheader("🧠Predict Churn")

# Slider for monthly charges (range 0 to 100, default 20)
monthly_charges = st.slider("Monthly_charges", 0, 100, 20)

# Slider for number of support calls (range 0 to 200, default 60)
support_calls = st.slider("Support_calls", 0, 200, 60)

# Slider for tenure in months (range 0 to 100, default 40)
tenure_months = st.slider("Tenure_months", 0, 100, 40)

# Dropdown selection for contract type, encoded as integers with labels
contract_type_encoded = st.selectbox(
    "Contract_type",
    [0, 1, 2],
    format_func=lambda x: ["Monthly", "1-Year", "2-Year"][x]
)

# Feature engineering:
# 1 if contract is longer than monthly, else 0
is_long_contract = 1 if contract_type_encoded > 0 else 0

# Calculate average support calls per month, add 1 to avoid division by zero
support_calls_per_month = support_calls / (tenure_months + 1)

# Calculate total revenue as tenure times monthly charges
revenue = tenure_months * monthly_charges

# Create a DataFrame for the model input with the engineered features
input_df = pd.DataFrame([{
    "TenureMonths": tenure_months,
    "MonthlyCharges": monthly_charges,
    "SupportCalls": support_calls,
    "ContractType_encoded": contract_type_encoded,
    "IsLongContract": is_long_contract,
    "Revenue": revenue,
    "SupportCallsPerMonth": support_calls_per_month
}])

# Define the exact feature order expected by the model
feature_order = [
    "TenureMonths",
    "MonthlyCharges",
    "SupportCalls",
    "ContractType_encoded",
    "IsLongContract",
    "Revenue",
    "SupportCallsPerMonth"
]

# Reorder input_df columns to match model's expected order
input_df = input_df[feature_order]

# When the Predict button is clicked:
if st.button("✅Predict Churn"):
    # Predict the class (0 = no churn, 1 = churn)
    prediction = model.predict(input_df)[0]

    # Predict the probability of churn (class 1)
    probability = model.predict_proba(input_df)[0][1]

    # Display the prediction result as text
    st.write(f"Prediction:\n {'churn' if prediction == 1 else 'not churn'}")

    # Display the probability of churn formatted to 2 decimal places
    st.write(f"Probability: {probability:.2f}")
