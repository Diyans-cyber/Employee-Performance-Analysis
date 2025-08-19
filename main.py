# =========================================
# Employee Performance Analysis
# Reads Employee.xls, cleans data, EDA, ML, Export
# =========================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 1) Load Excel File
# -------------------------------
excel_path = "Employee.xls"   # Change path if needed
sheet_name = 0                # First sheet

if not os.path.exists(excel_path):
    raise FileNotFoundError(f"File not found: {excel_path}")

df = pd.read_excel(excel_path, sheet_name=sheet_name)

print(f"\n✅ Loaded dataset with shape: {df.shape}")
print("\n--- Column names in dataset ---")
print(df.columns.tolist())

# -------------------------------
# 2) Rename Columns Manually (based on your dataset)
# ⚠️ Adjust these mappings after checking column names printed above
# -------------------------------
rename_map = {
    "EmpNumber": "employee_id",
    "Age": "age",
    "Gender": "gender",
    "EmpDepartment": "department",
    "TotalWorkExperienceInYears": "years_experience",
    "TrainingTimesLastYear": "training_hours",
    "PerformanceRating": "performance_score"
}


# Apply renaming where possible
df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

print("\n--- Columns after renaming ---")
print(df.columns.tolist())

# -------------------------------
# 3) Keep Relevant Columns
# -------------------------------
keep_cols = [c for c in [
    "employee_id", "age", "gender", "department",
    "years_experience", "training_hours", "salary", "performance_score"
] if c in df.columns]

df = df[keep_cols].copy()
print(f"\n✅ Using columns: {keep_cols}")

# -------------------------------
# 4) Handle Missing Values
# -------------------------------
for col in ["years_experience", "training_hours", "age", "salary"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

for col in ["gender", "department"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
        df[col].fillna("Unknown", inplace=True)

# Clean performance score
if "performance_score" in df.columns:
    perf = df["performance_score"].astype(str).str.strip().str.lower()
    label_map = {
        "very low": 1, "low": 2, "average": 3, "medium": 3,
        "mid": 3, "high": 4, "very high": 5, "excellent": 5, "poor": 2, "good": 4
    }
    perf_num = pd.to_numeric(perf, errors="coerce")
    if perf_num.isna().mean() < 0.5:
        perf_num = perf_num.clip(1, 5)
        df["performance_score"] = perf_num.astype(int)
    else:
        df["performance_score"] = perf.map(label_map).fillna(3).astype(int)
    df["performance_score"] = df["performance_score"].clip(1, 5)

print("\n✅ Data cleaned. Sample rows:")
print(df.head())

# -------------------------------
# 5) Exploratory Data Analysis
# -------------------------------
sns.set_theme()

if "training_hours" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="performance_score", y="training_hours", data=df)
    plt.title("Training Hours vs Performance")
    plt.show()

if "years_experience" in df.columns and "department" in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="years_experience", y="performance_score",
                    hue="department", data=df, alpha=0.7)
    plt.title("Experience vs Performance by Department")
    plt.show()

if "department" in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x="department", y="performance_score", data=df, estimator=np.mean)
    plt.title("Average Performance by Department")
    plt.xticks(rotation=20)
    plt.show()

# -------------------------------
# 6) Machine Learning Model
# -------------------------------
if "performance_score" in df.columns:
    features = [c for c in ["training_hours", "years_experience", "age", "salary", "department", "gender"] if c in df.columns]
    X = df[features].copy()
    y = df["performance_score"]

    # Encode categoricals
    for col in ["department", "gender"]:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Feature Importance
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n--- Feature Importance ---")
    print(importance)

    plt.figure(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=importance)
    plt.title("Feature Importance in Predicting Performance")
    plt.show()

# -------------------------------
# 7) Save Cleaned Data for Power BI
# -------------------------------
df.to_csv("employee_clean.csv", index=False)
print("\n✅ Exported cleaned dataset -> employee_clean.csv (use in Power BI)")
