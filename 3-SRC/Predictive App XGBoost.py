import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ----------------------- DATA PREPROCESSING & MODEL TRAINING -----------------------

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
    df[col] = le.fit_transform(df[col])  # Convert "Yes/No" to 1/0
    label_encoders[col] = le  # Save encoder for future use

# Apply One-Hot Encoding for Multi-category Columns
df = pd.get_dummies(df, columns=["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"], drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Save feature names (for aligning inference data)
feature_names = X.columns.tolist()
joblib.dump(feature_names, "4-MODELS/Feature_Names.pkl")

# Standardize numeric features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Evaluate Models
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("\nXGBoost Report:\n", classification_report(y_test, xgb_pred))

# Save trained model & preprocessors
joblib.dump(xgb_model, "4-MODELS/Attrition_Model.pkl")
joblib.dump(scaler, "4-MODELS/Scaler.pkl")
joblib.dump(label_encoders, "4-MODELS/Label_Encoders.pkl")

# ----------------------- GRADIO APP -----------------------

# Load trained model & preprocessors
xgb_model = joblib.load("4-MODELS/Attrition_Model.pkl")
scaler = joblib.load("4-MODELS/Scaler.pkl")
feature_names = joblib.load("4-MODELS/Feature_Names.pkl")

# Load images for visualization
stay_image = Image.open("2-ASSETS/Employee_Stay.jpg")  # Ensure the image file exists
leave_image = Image.open("2-ASSETS/Employee_Leave.jpg")  # Ensure the image file exists

# Define valid input ranges
limits = {
    "Age": (18, 60),
    "Distance From Home": (1, 29),
    "Monthly Income": (1009, 19999),
    "Years At Company": (0, 40),
    "Years Since Last Promotion": (0, 15),
    "Environment Satisfaction": (1, 4),
    "Percent Salary Hike": (11, 25),
    "Work Life Balance": (1, 4),
    "Job Satisfaction": (1, 4),
}

# Define Gradio Inputs
inputs = [
    gr.Slider(label="Age", minimum=limits["Age"][0], maximum=limits["Age"][1], value=30, step=1),
    gr.Slider(label="Distance From Home", minimum=limits["Distance From Home"][0], maximum=limits["Distance From Home"][1], value=5, step=1),
    gr.Slider(label="Monthly Income", minimum=limits["Monthly Income"][0], maximum=limits["Monthly Income"][1], value=5000, step=500),
    gr.Slider(label="Years At Company", minimum=limits["Years At Company"][0], maximum=limits["Years At Company"][1], value=3, step=1),
    gr.Slider(label="Years Since Last Promotion", minimum=limits["Years Since Last Promotion"][0], maximum=limits["Years Since Last Promotion"][1], value=1, step=1),
    gr.Slider(label="Environment Satisfaction (1-4)", minimum=limits["Environment Satisfaction"][0], maximum=limits["Environment Satisfaction"][1], value=3, step=1),
    gr.Slider(label="Percentage Salary Hike", minimum=limits["Percent Salary Hike"][0], maximum=limits["Percent Salary Hike"][1], value=15, step=1),
    gr.Slider(label="Work Life Balance (1-4)", minimum=limits["Work Life Balance"][0], maximum=limits["Work Life Balance"][1], value=3, step=1),
    gr.Slider(label="Job Satisfaction (1-4)", minimum=limits["Job Satisfaction"][0], maximum=limits["Job Satisfaction"][1], value=3, step=1),
    gr.Dropdown(["Yes", "No"], label="OverTime", value="No"),
    gr.Dropdown(["Male", "Female"], label="Gender", value="Male") 
]

outputs = [gr.Label(label="Attrition Prediction"), gr.Image(type="pil", label="Result Image")]

# Prediction Function
def predict_attrition(age, distance, income, years_at_company, years_since_last_promotion, environment_satisfaction,
                      percentage_salary_hike, work_life_balance, job_satisfaction, overtime, gender):
    try:
        # Convert Gender to numeric
        gender_numeric = 1 if gender == "Male" else 0  # Male = 1, Female = 0

        # Convert OverTime to numeric
        overtime_numeric = 1 if overtime == "Yes" else 0

        # Prepare input data
        input_data = pd.DataFrame({
            "Age": [age],
            "DistanceFromHome": [distance],
            "MonthlyIncome": [income],
            "YearsAtCompany": [years_at_company],
            "YearsSinceLastPromotion": [years_since_last_promotion],
            "EnvironmentSatisfaction": [environment_satisfaction],  
            "PercentSalaryHike": [percentage_salary_hike],  
            "WorkLifeBalance": [work_life_balance],
            "JobSatisfaction": [job_satisfaction],
            "OverTime": [overtime_numeric],
            "Gender": [gender_numeric]
        })

        # Align with original training feature names (ensures missing columns are filled with 0)
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = xgb_model.predict(input_scaled)

        # Return Result
        if prediction == 1:
            return "Likely to Leave", leave_image
        else:
            return "Likely to Stay", stay_image

    except Exception as e:
        return str(e), None

# Gradio App
app = gr.Interface(
    fn=predict_attrition,
    inputs=inputs,
    outputs=outputs,
    description="This application predicts employee attrition using machine learning. "
                "Enter employee details to determine whether they are likely to stay or leave.",
    title="Employee Attrition Prediction"
)

app.launch(share=True)