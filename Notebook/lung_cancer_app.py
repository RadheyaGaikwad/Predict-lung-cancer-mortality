import streamlit as st
import pickle
import pandas as pd

# ✅ Load the trained pipeline
with open('Notebook/model.pkl', 'rb') as f:
    model = pickle.load(f)

# 🎉 App UI
st.set_page_config(page_title="🫁 Lung Cancer Prediction", page_icon="🫁", layout="centered")

st.title("🫁 Lung Cancer Prediction App")
st.write("""
This app predicts the **likelihood of lung cancer** based on patient information.
Please provide your details on the sidebar 👉.
""")

# ✅ Sidebar inputs
st.sidebar.header("📋 User Input Features")

age = st.sidebar.slider("Age", 0, 120, 30)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    ['Passive Smoker', 'Former Smoker', 'Never Smoked', 'Current Smoker']
)
cancer_stage = st.sidebar.selectbox(
    "Cancer Stage",
    ['Stage I', 'Stage II', 'Stage III', 'Stage IV']  # only valid stages
)
family_history = st.sidebar.selectbox("Family History of Lung Cancer", ['Yes', 'No'])
treatment_type = st.sidebar.selectbox(
    "Treatment Type",
    ['Chemotherapy', 'Surgery', 'Combined', 'Radiation']
)

# ✅ Add BMI with meaningful label
bmi = st.sidebar.number_input(
    "Body Mass Index (BMI — a measure of body fat based on weight & height)",
    min_value=10.0,
    max_value=50.0,
    value=22.0,
    step=0.1
)

# ✅ Encode binary columns to match pipeline passthrough
gender_num = 1 if gender == 'Male' else 0
family_history_num = 1 if family_history == 'Yes' else 0

# ✅ Build input DataFrame exactly like training
input_df = pd.DataFrame({
    'smoking_status': [smoking_status],
    'treatment_type': [treatment_type],
    'cancer_stage': [cancer_stage],
    'age': [age],
    'bmi': [bmi],
    'gender': [gender_num],
    'family_history': [family_history_num]
})

st.write("### ✔️ Your Input:")
st.write(input_df)

# ✅ Predict button
if st.button('🔍 Predict Lung Cancer Risk'):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Lung Cancer Detected!** Please consult a doctor.")
    else:
        st.success("✅ **Low Risk of Lung Cancer!** Stay healthy and take preventive care.")


