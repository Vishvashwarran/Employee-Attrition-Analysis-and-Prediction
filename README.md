📊 Employee Attrition Analysis & Prediction
🔍 Overview

This project focuses on analyzing employee attrition patterns and predicting whether an employee is likely to leave the organization.
It uses Machine Learning (Logistic Regression) with SMOTE, performs full EDA, and includes a Streamlit web application for real-time prediction.

🎯 Objectives

Understand factors influencing employee attrition
Build ML models to accurately predict attrition
Handle class imbalance using SMOTE
Develop a Streamlit dashboard for predictions
Provide a second ML prediction (Performance Rating)
Present clear visualizations and business insights

🧠 Skills Demonstrated

Data Cleaning & Preprocessing
Feature Engineering
Exploratory Data Analysis (EDA)
Classification Models
SMOTE Oversampling
Model Evaluation Metrics
Streamlit App Development
GitHub Version Control

📁 Project Structure
employee-attrition-project/
│
├── app.py                   # Streamlit web app
├── main.ipynb               # Full ML pipeline (Colab/Jupyter)
├── best_model.joblib        # Trained attrition prediction model
├── Employee-Attrition.csv   # Dataset
├── requirements.txt         # Dependencies
└── README.md                # Documentation

🧹 Data Preprocessing

Removed unnecessary columns: EmployeeCount, Over18, StandardHours, EmployeeNumber
Outliers clipped using 5th–95th percentile
Label encoding + OneHotEncoding
Feature Scaling using StandardScaler
SMOTE applied for class imbalance
Train/Test split with stratification

🔬 Exploratory Data Analysis (EDA)

Visualizations included:

Attrition Count Plot
Gender vs Attrition
Job Role vs Attrition
Correlation Heatmap
Job Satisfaction vs Attrition
Monthly Income vs Attrition
Years at Company vs Attrition

These help understand key factors influencing attrition.

🤖 Machine Learning Models

Tested models:

Logistic Regression
Decision Tree
Random Forest
KNN
Bagging Classifier
AdaBoost
Gradient Boosting

✔ Best model selected using GridSearchCV
✔ Logistic Regression with class weight tuning
✔ Model saved as best_model.joblib

🧪 Model Evaluation
Attrition Prediction

Metrics used:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
AUC-ROC Curve


🎮 Streamlit App

The web app includes:

Attrition visualizations
Employee detail input form
Real-time attrition prediction
Probability score
Clean and responsive layout



🔮 Use Cases

Identify high-risk employees
Support HR retention strategies
Understand attrition factors
Predict performance rating for employee evaluations

👤 Author

Vishvashwarran V B
