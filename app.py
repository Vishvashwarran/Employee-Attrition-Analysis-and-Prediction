import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("📊 Employee Attrition Dashboard & Prediction App")


df = pd.read_csv("cleaned_employee_data.csv")


model = load("best_model.joblib")
perf_model = load("performance_model.joblib")


tab1, tab2, tab3 = st.tabs(["📈 Visualizations", "🔮 Attrition Prediction", "⭐ Performance Rating Prediction"])



with tab1:

    st.header("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    # 1. Attrition Count Plot
    with col1:
        st.subheader("Attrition Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Attrition', data=df, palette='coolwarm', ax=ax1)
        st.pyplot(fig1)

    # 2. Gender vs Att
    with col2:
        st.subheader("Gender vs Attrition")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Gender', hue='Attrition', data=df, palette='Set2', ax=ax2)
        st.pyplot(fig2)

    # 3. Job Role vs Att
    st.subheader("JobRole vs Attrition")
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.countplot(x='JobRole', hue='Attrition', data=df, palette='husl', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # 4. Heatmap
    st.subheader("Correlation Heatmap")

    # 4. Correlation Heatmap (Full Matrix + Values)
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    fig4, ax4 = plt.subplots(figsize=(20, 12))
    sns.heatmap(numeric_df.corr(),cmap='coolwarm',annot=True, fmt=".2f",ax=ax4)
    st.pyplot(fig4)


    #5. Job sat vs Att

    st.subheader("Job Staisfaction vs Attrition")

    fig5, ax5 = plt.subplots(figsize=(6,4))
    sns.barplot(x='JobSatisfaction', y='Attrition', data=df, orient='h', ax=ax5)
    st.pyplot(fig5)

    #6 Salary vs Att

    st.subheader("Monthly Income vs Attrition")

    fig6, ax6 = plt.subplots(figsize=(6,4))
    sns.barplot(y='MonthlyIncome', x='Attrition', data=df, ax=ax6)
    st.pyplot(fig6)

    #7 Years at comp

    st.subheader("Years at Company vs Attrition")

    fig7, ax7 = plt.subplots(figsize=(6,4))
    sns.barplot(y='Attrition', x='YearsAtCompany', data=df, orient='h')
    st.pyplot(fig7)




with tab2:

    st.header("🔮 Predict Employee Attrition")

    with st.form("attrition_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            Age = st.number_input("Age", 18, 60, 30)
            Gender_input = st.selectbox("Gender", ["Male", "Female"])
            MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
        with col2:
            DistanceFromHome = st.number_input("Distance From Home (KM)", 0, 50, 5)
            NumCompaniesWorked = st.number_input("NumCompaniesWorked", 0, 20, 3)
            OverTime_input = st.selectbox("OverTime", ["Yes", "No"])

        st.subheader("Job Information")
        col3, col4 = st.columns(2)
        with col3:
            Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            JobRole = st.selectbox("JobRole", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"])
            BusinessTravel = st.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            JobLevel = st.number_input("JobLevel (1–5)", 1, 5, 2)
            st.caption("1 = Entry Level, 2 = Mid Level, 3 = Senior Level, 4 = Manager, 5 = Executive")
        with col4:
            YearsAtCompany = st.number_input("YearsAtCompany", 0, 40, 5)
            YearsInCurrentRole = st.number_input("YearsInCurrentRole", 0, 20, 3)
            YearsWithCurrManager = st.number_input("YearsWithCurrManager", 0, 20, 3)
            YearsSinceLastPromotion = st.number_input("YearsSinceLastPromotion", 0, 20, 2)

        st.subheader("Salary Details")
        col5, col6 = st.columns(2)
        with col5:
            MonthlyIncome = st.number_input("MonthlyIncome", 1000, 20000, 5000)
            MonthlyRate = st.number_input("MonthlyRate", 1000, 30000, 10000)
            HourlyRate = st.number_input("HourlyRate", 10, 200, 50)
        with col6:
            DailyRate = st.number_input("DailyRate", 100, 2000, 800)
            PercentSalaryHike = st.number_input("PercentSalaryHike", 0, 50, 15)
            StockOptionLevel = st.number_input("StockOptionLevel (0–3)", 0, 3, 1)
            st.caption("0 = No Options, 1 = Low, 2 = Medium, 3 = High")


        st.subheader("Satisfaction Scores (1–4)")
        st.caption("Scale: 1 = Low, 2 = Medium, 3 = High, 4 = Very High")
        col7, col8 = st.columns(2)
        with col7:
            JobSatisfaction = st.number_input("JobSatisfaction", 1, 4, 3)
            EnvironmentSatisfaction = st.number_input("EnvironmentSatisfaction", 1, 4, 3)

            RelationshipSatisfaction = st.number_input("RelationshipSatisfaction", 1, 4, 3)
            st.caption("1 = Low, 2 = Medium, 3 = High, 4 = Very High")
        with col8:
            WorkLifeBalance = st.number_input("WorkLifeBalance", 1, 4, 3)
            st.caption("1 = Low, 2 = Medium, 3 = High, 4 = Very High")
            JobInvolvement = st.number_input("JobInvolvement", 1, 4, 3)
            PerformanceRating = st.number_input("Performance Rating (1–4)", 1, 4, 3)
            st.caption("1 = Low, 2 = Good, 3 = Excellent, 4 = Outstanding")



        st.subheader("Training & Education")
        Education = st.number_input("Education (1–5)", 1, 5, 3)
        st.caption("1 = Below College, 2 = College, 3 = Bachelor, 4 = Master, 5 = Doctorate")
        EducationField = st.selectbox("EducationField", [
            "Life Sciences", "Medical", "Marketing",
            "Technical Degree", "Human Resources", "Other"])
        TrainingTimesLastYear = st.number_input("TrainingTimesLastYear", 0, 10, 3)
        TotalWorkingYears = st.number_input("TotalWorkingYears", 0, 40, 5)

        submit = st.form_submit_button("Predict")

    if submit:

        Gender = 1 if Gender_input == "Male" else 0
        OverTime = 1 if OverTime_input == "Yes" else 0

        input_data = pd.DataFrame([{
            "Age": Age,
            "BusinessTravel": BusinessTravel,
            "DailyRate": DailyRate,
            "Department": Department,
            "DistanceFromHome": DistanceFromHome,
            "Education": Education,
            "EducationField": EducationField,
            "EnvironmentSatisfaction": EnvironmentSatisfaction,
            "Gender": Gender,
            "HourlyRate": HourlyRate,
            "JobInvolvement": JobInvolvement,
            "JobLevel": JobLevel,
            "JobRole": JobRole,
            "JobSatisfaction": JobSatisfaction,
            "PerformanceRating": PerformanceRating,
            "MaritalStatus": MaritalStatus,
            "MonthlyIncome": MonthlyIncome,
            "MonthlyRate": MonthlyRate,
            "NumCompaniesWorked": NumCompaniesWorked,
            "OverTime": OverTime,
            "PercentSalaryHike": PercentSalaryHike,
            "RelationshipSatisfaction": RelationshipSatisfaction,
            "StockOptionLevel": StockOptionLevel,
            "TotalWorkingYears": TotalWorkingYears,
            "TrainingTimesLastYear": TrainingTimesLastYear,
            "WorkLifeBalance": WorkLifeBalance,
            "YearsAtCompany": YearsAtCompany,
            "YearsInCurrentRole": YearsInCurrentRole,
            "YearsSinceLastPromotion": YearsSinceLastPromotion,
            "YearsWithCurrManager": YearsWithCurrManager
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠ HIGH Attrition Risk (Probability: {probability:.2f})")
        else:
            st.success(f"✔ LOW Attrition Risk (Probability: {probability:.2f})")


#Performance Prediction


with tab3:

    st.header("⭐ Predict Employee Performance Rating")

    with st.form("performance_form"):
        st.subheader("Employee Features")

        colp1, colp2 = st.columns(2)

        with colp1:
            Age_p = st.number_input("Age", 18, 60, 30)
            JobInvolvement_p = st.number_input("JobInvolvement (1-4)", 1, 4, 3)
            JobLevel_p = st.number_input("JobLevel (1-5)", 1, 5, 2)
            st.caption("1 = Entry Level, 2 = Mid Level, 3 = Senior Level, 4 = Manager, 5 = Executive")
            MonthlyIncome_p = st.number_input("MonthlyIncome", 1000, 20000, 5000)

        with colp2:
            TotalWorkingYears_p = st.number_input("TotalWorkingYears", 0, 40, 5)
            YearsAtCompany_p = st.number_input("YearsAtCompany", 0, 40, 5)
            YearsInCurrentRole_p = st.number_input("YearsInCurrentRole", 0, 20, 3)
            Education_p = st.number_input("Education (1–5)", 1, 5, 3)
            st.caption("1 = Below College, 2 = College, 3 = Bachelor, 4 = Master, 5 = Doctorate")

        JobRole_p = st.selectbox("JobRole", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"])

        Gender_p = st.selectbox("Gender", ["Male", "Female"])
        Gender_p = 1 if Gender_p == "Male" else 0

        EducationField_p = st.selectbox("EducationField", [
            "Life Sciences", "Medical", "Marketing",
            "Technical Degree", "Human Resources", "Other"])

        submit_p = st.form_submit_button("Predict Performance Rating")

    if submit_p:
        input_data2 = pd.DataFrame([{
            "Age": Age_p,
            "JobInvolvement": JobInvolvement_p,
            "JobLevel": JobLevel_p,
            "MonthlyIncome": MonthlyIncome_p,
            "TotalWorkingYears": TotalWorkingYears_p,
            "YearsAtCompany": YearsAtCompany_p,
            "YearsInCurrentRole": YearsInCurrentRole_p,
            "Education": Education_p,
            "JobRole": JobRole_p,
            "Gender": Gender_p,
            "EducationField": EducationField_p
        }])

        prediction2 = perf_model.predict(input_data2)[0]

        st.success(f"⭐ Predicted Performance Rating: {prediction2}")
