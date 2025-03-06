import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("cvd_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# App Title
st.title("CVD Risk Prediction")

# Numeric Inputs
age = st.number_input("Age", min_value=20, max_value=100, step=1)
fasting_bg_mgdl = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=50, max_value=300)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150)
cholesterol_mgdl = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400)
bp_systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200)
bp_diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=120)

# Categorical Inputs with One-Hot Encoding
sex_map = {"Male": 0, "Female": 1}
sex = st.selectbox("Sex", list(sex_map.keys()))

edu_level_map = {"No formal education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
edu_level = st.selectbox("Education Level", list(edu_level_map.keys()))

monthly_income_map = {"Low": 0, "Medium": 1, "High": 2}
monthly_income = st.selectbox("Monthly Income", list(monthly_income_map.keys()))

alcohol_map = {"No": 0, "Occasionally": 1, "Frequently": 2}
alcohol_consumption = st.selectbox("Alcohol Consumption", list(alcohol_map.keys()))

diet_quality_map = {"Poor": 0, "Average": 1, "Good": 2}
diet_quality = st.selectbox("Diet Quality", list(diet_quality_map.keys()))

salt_intake_map = {"Low": 0, "Moderate": 1, "High": 2}
salt_intake = st.selectbox("Salt Intake", list(salt_intake_map.keys()))

physical_activity_map = {"Low": 0, "Moderate": 1, "High": 2}
physical_activity = st.selectbox("Physical Activity Level", list(physical_activity_map.keys()))

bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
bmi_category = st.selectbox("BMI Category", list(bmi_map.keys()))

# One-Hot Encoded Categorical Variables
work_mapping = {
    "Unemployed": [1, 0, 0, 0, 0, 0, 0], "Self-employed": [0, 1, 0, 0, 0, 0, 0],
    "Government": [0, 0, 1, 0, 0, 0, 0], "Private": [0, 0, 0, 1, 0, 0, 0],
    "Retired": [0, 0, 0, 0, 1, 0, 0], "Other": [0, 0, 0, 0, 0, 1, 0], "Freelancer": [0, 0, 0, 0, 0, 0, 1]
}
work = st.selectbox("Work Type", list(work_mapping.keys()))

tobacco_mapping = {
    "Non-User": [1, 0, 0], "Past User": [0, 1, 0], "Current User": [0, 0, 1]
}
tobacco_use = st.selectbox("Tobacco Use", list(tobacco_mapping.keys()))

bp_status_mapping = {
    "Have Raised Blood Pressure": [1, 0, 0], "Haven't Checked": [0, 1, 0], "No Raised Blood Pressure": [0, 0, 1]
}
bp_status = st.selectbox("Blood Pressure Status", list(bp_status_mapping.keys()))

diabetes_mapping = {
    "Have Diabetes": [1, 0, 0], "Haven't Checked": [0, 1, 0], "No Diabetes": [0, 0, 1]
}
diabetes_status = st.selectbox("Diabetes Status", list(diabetes_mapping.keys()))

cholesterol_mapping = {
    "Have Raised Cholesterol": [1, 0, 0], "Haven't Checked": [0, 1, 0], "No Raised Cholesterol": [0, 0, 1]
}
cholesterol_status = st.selectbox("Cholesterol Status", list(cholesterol_mapping.keys()))

# Combine Inputs into Final Feature Vector
input_data = np.array([[age, sex_map[sex], edu_level_map[edu_level], monthly_income_map[monthly_income], 
                         alcohol_map[alcohol_consumption], diet_quality_map[diet_quality], salt_intake_map[salt_intake], 
                         physical_activity_map[physical_activity], heart_rate, bp_systolic, bp_diastolic, 
                         cholesterol_mgdl, fasting_bg_mgdl, bmi_map[bmi_category]] + 
                        work_mapping[work] + tobacco_mapping[tobacco_use] + bp_status_mapping[bp_status] +
                        diabetes_mapping[diabetes_status] + cholesterol_mapping[cholesterol_status]])

# Predict Button
if st.button("Predict CVD Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of having CVD

    if prediction == 1:
        st.error(f"🚨 Based on the provided information, there may be a higher risk of developing cardiovascular disease. Consider consulting a healthcare professional for further evaluation! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Based on the provided information, there are no strong indications og high cardiovascular risk. Maintaining a health lifestyle is still important! Probability: {probability:.2f}")
