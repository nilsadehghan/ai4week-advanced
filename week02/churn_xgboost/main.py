import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score, roc_curve, auc
from sklearn.utils import class_weight
import numpy as np
from imblearn.over_sampling import SMOTE

# Function for feature engineering on the dataset
def feature_engineering(df):
    # Drop rows with any missing values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create a new binary feature: 'Frequent caller' = 'Yes' if SupportCalls > 2, else 'No'
    df["Frequent caller"] = df["SupportCalls"].apply(lambda x: "Yes" if x > 2 else "No")

    # Encode 'ContractType' feature if it exists
    if "ContractType" in df.columns:
        le = LabelEncoder()
        # Convert categorical 'ContractType' to numeric labels
        df["ContractType_encoded"] = le.fit_transform(df["ContractType"])
        # Drop original 'ContractType' column after encoding
        df.drop("ContractType", axis=1, inplace=True)

    # Create a new binary feature 'IsLongContract': 1 if encoded ContractType > 0, else 0
    df["IsLongContract"] = df["ContractType_encoded"].apply(lambda x: 1 if x > 0 else 0)

    # Calculate total revenue as tenure multiplied by monthly charges
    df["Revenue"] = df["TenureMonths"] * df["MonthlyCharges"]

    # Calculate support calls per month (to normalize by tenure)
    df["SupportCallsPerMonth"] = df["SupportCalls"] / (df["TenureMonths"] + 1)

    return df

# Load the dataset
df = pd.read_csv("synthetic_churn_data.csv")

# Apply feature engineering
df = feature_engineering(df)

# Save the processed dataset to a new CSV file
df.to_csv("synthetic_churn_data_featured.csv", index=False)

# Display first few rows of the processed data
print(df.head())

# Print dataset shape (rows, columns)
print(df.shape)

# Check for any remaining missing values
print(df.isnull().sum())

# Get descriptive statistics for numeric columns
print(df.describe())

# Count total number of entries in 'Churn' column
print(df['Churn'].count())

# Visualize distribution of MonthlyCharges by churn status
plt.title("MonthlyCharges vs Churn")
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.show()

# Visualize distribution of SupportCalls by churn status
plt.title("SupportCalls vs Churn")
sns.boxplot(x="Churn", y="SupportCalls", data=df)
plt.show()

# Visualize distribution of IsLongContract by churn status
plt.title("IsLongContract vs Churn")
sns.boxplot(x="Churn", y="IsLongContract", data=df)
plt.show()

# Visualize distribution of SupportCallsPerMonth by churn status
plt.title("SupportCallsPerMonth vs Churn")
sns.boxplot(x="Churn", y="SupportCallsPerMonth", data=df)
plt.show()

# Visualize distribution of Revenue by churn status
plt.title("Revenue vs Churn")
sns.boxplot(x="Churn", y="Revenue", data=df)
plt.show()

# Print counts of each class in the target variable 'Churn'
print(df["Churn"].value_counts())

# Plot count of each ContractType_encoded category split by churn status
sns.countplot(x="ContractType_encoded", hue="Churn", data=df)
plt.show()

# Prepare features and target variable for modeling
# Drop columns not used as features
df_filtered = df.drop(columns=["Churn", "CustomerID", "SignupDate", "LastPurchaseDate", "Frequent caller"])

x = df_filtered  # Features
y = df["Churn"]  # Target

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
sm = SMOTE(random_state=42)
x_resampled, y_resampled = sm.fit_resample(x, y)

# Split the resampled dataset into training and testing sets (80-20 split)
X_train, X_test, Y_train, Y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# Compute class weights to handle class imbalance during training
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y)
# Calculate scale_pos_weight for XGBoost as ratio of negative to positive class weights
scale_pos_weight = weights[0] / weights[1]

# Initialize XGBoost classifier with imbalance parameter and evaluation metric
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=scale_pos_weight)

# Train the model on the training data
model.fit(X_train, Y_train)

# Predict churn on the test data
y_predict = model.predict(X_test)

# Calculate evaluation metrics
precision = precision_score(Y_test, y_predict, average="binary")
recall = recall_score(Y_test, y_predict, average="binary")
accuracy = accuracy_score(Y_test, y_predict)
f1 = f1_score(Y_test, y_predict)

# Print model performance metrics
print(f"Accuracy_score: {accuracy:.4f}")
print(f"F1_score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Classification_report:")
print(classification_report(Y_test, y_predict))

# Compute predicted probabilities for ROC curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
auc_result = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label=f"AUC = {auc_result:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="red")  # Diagonal line (random guess)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Save the trained XGBoost model to file
model.save_model("xgb_model.json")
