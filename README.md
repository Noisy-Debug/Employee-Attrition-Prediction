# 🧠 Employee Attrition Prediction System

This project implements a machine learning-based predictive system to forecast employee attrition. It uses the **XGBoost classifier**, trained on historical HR data, to help organizations proactively retain top talent. It also includes a **Gradio web application** to simulate predictions based on user inputs.

# 📊 Problem Statement

Traditional HR approaches like exit interviews are reactive and provide little time to prevent turnover. This project aims to **predict attrition early** by analyzing patterns in employee data.

# 🔍 Key Features

- **4 ML Models Compared**: Logistic Regression, Decision Tree, Random Forest, and XGBoost (see: 3-SRC/ML Models Testing.py)
- **Best Accuracy**: XGBoost (92%).
- **Live Predictions**: Built with Gradio to visualize likely outcomes.
- **Insightful Charts**: Understand which roles, departments, and conditions contribute to attrition.

# 💡 Technologies Used

- Python, Pandas, Scikit-learn, XGBoost
- Gradio (for Web App)
- Joblib (for model saving)
- Matplotlib / Seaborn (for visualization)

# 📦 Repository Contents

| Folder          | Description |
|-----------------|-------------|
| `DATA/`         | HR dataset used for training. |
| `ASSETS/`       | Visual indicators for stay/leave prediction. |
| `SRC/`          | Contains model training and app deployment code. |
| `MODELS/`       | Saved models and encoders. |
| `NOTEBOOK/`     | EDA and visualization notebook. |
| `REPORT/`       | Project report and presentation slides. |

# 🚀 Running the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/Noisy-Debug/Employee-Attrition-Prediction.git

# 2. Install dependencies
pip install -r REQUIREMENTS.txt

# 3. Run Gradio app
python 3-SRC/Predictive App XGBoost.py
```
# 🌐 Try It on Hugging Face
Don't want to run it locally? Try the live demo here: https://huggingface.co/spaces/Noisy-Debug/Employee-Attrition-Prediction-App

# 📊 Results and Insights

- XGBoost achieved 92% accuracy
- Employees with low income, poor work-life balance, and low job satisfaction were more likely to leave
- Highest attrition observed in Sales and R&D
- Roles most at risk are Sales Executives and Laboratory Technicians

For visual charts, see: 5-NOTEBOOK/Visulization.ipynb

# 🔮 Future Enhancements

- Implement Explainable AI (XAI) methods
- Add bias/fairness analysis for sensitive features
- Expand dataset across multiple organizations
