# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "1-DATA/HR-Employee-Attrition.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df.drop(columns=["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"], inplace=True)

# Encode categorical variables
binary_cols = ["Attrition", "OverTime", "Gender"]
label_encoders = {}

for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert to 0/1
    label_encoders[col] = le

# Apply One-Hot Encoding for multi-category columns
df = pd.get_dummies(df, columns=["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"], drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Standardize numeric features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split into train & test sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

# Train Decision Tree Model
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Evaluate Models
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("\nLogistic Regression Report:\n", classification_report(y_test, log_reg_pred))
print("\nDecision Tree Report:\n", classification_report(y_test, dt_pred))
print("\nRandom Forest Report:\n", classification_report(y_test, rf_pred))
print("\nXGBoost Report:\n", classification_report(y_test, xgb_pred))