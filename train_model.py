import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("dataset/loan.csv")

print("Dataset Loaded Successfully")
print(df.head())
print(df.info())

print("\n==============================")
print("Missing Values")
print("==============================")

print(df.isnull().sum())

# Fill categorical missing values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

# Fill numerical missing values
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==============================
# DATA VISUALIZATION - BOX PLOT
# ==============================

plt.figure(figsize=(10,6))

sns.boxplot(x=df["LoanAmount"])

plt.title("Loan Amount Box Plot")

plt.savefig("visualizations/boxplot.png")
plt.show()

# ==========================
# Distribution Plot
# ==========================

plt.figure(figsize=(10,6))
sns.histplot(df["LoanAmount"], kde=True)
plt.title("Loan Amount Distribution")
plt.savefig("visualizations/distribution_plot.png")
plt.close()

# ==========================
# Applicant Income Box Plot
# ==========================

plt.figure(figsize=(10,6))
sns.boxplot(x=df["ApplicantIncome"])
plt.title("Applicant Income Quantile Plot")
plt.savefig("visualizations/quantile_plot.png")
plt.close()

# ==========================
# Correlation Heatmap
# ==========================

numeric_df = df.select_dtypes(include=["int64","float64"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("visualizations/heatmap.png")
plt.close()

print("Visualizations Saved Successfully")

# ======================================
# Encode Categorical Columns
# ======================================

label_encoders = {}

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Drop Loan_ID if present
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# ======================================
# Features and Target
# ======================================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# ======================================
# Feature Scaling
# ======================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# Train Models
# ======================================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

best_model = None
best_accuracy = 0

print("\n========== MODEL ACCURACY ==========\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name}: {acc*100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("\nBest Model Accuracy:", round(best_accuracy*100,2), "%")

# ======================================
# Save Model
# ======================================

joblib.dump(best_model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("\nModel Saved Successfully!")