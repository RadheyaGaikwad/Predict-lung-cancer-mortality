import streamlit as st
import pickle
import pandas as pd

# 📌 Load your trained pipeline
model = pickle.load(open('Notebook/model.pkl', 'rb'))

# 🎉 App Title & Description
st.set_page_config(page_title="💙 Lung Cancer Prediction", page_icon="🫁", layout="centered")

st.title("🫁 Lung Cancer Prediction App")
st.write("""
Welcome! This app predicts the **likelihood of lung cancer** based on your input information.
Please provide honest and accurate details for best results.
""")

#  Sidebar for inputs
st.sidebar.header(" User Input Features")

def user_input_features():
    age = st.sidebar.number_input(" Age", min_value=0, max_value=120, value=30)

    gender = st.sidebar.selectbox(' Gender', ['Male', 'Female'])

    smoking_status = st.sidebar.selectbox(
        'Smoking Status',
        ['Passive Smoker', 'Former Smoker', 'Never Smoked', 'Current Smoker']
    )

    cancer_stage = st.sidebar.selectbox(
        'Cancer Stage',
        ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    )

    family_history = st.sidebar.selectbox(
        'Family History of Lung Cancer',
        ['Yes', 'No']
    )

    treatment_type = st.sidebar.selectbox(
        '💉 Treatment Type',
        ['Surgery', 'Chemotherapy', 'Radiation Therapy', 'Targeted Therapy']
    )

    # ✅ Map binary inputs to numeric
    gender = 1 if gender == 'Male' else 0
    family_history = 1 if family_history == 'Yes' else 0

    # ✅ Combine into a DataFrame
    features = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'smoking_status': [smoking_status],
        'cancer_stage': [cancer_stage],
        'family_history': [family_history],
        'treatment_type': [treatment_type]
    })

    return features

features_df = user_input_features()

# Show user input
st.subheader("🔍 Your Input Data")
st.write(features_df)

# Prediction Button


if st.button('💡 Predict Lung Cancer Survival'):
    prediction = model.predict(features_df)

    if prediction[0] == 1:
        st.error("⚠️ **Survival Chances Are Lower:** The model predicts a higher risk. Please consult a doctor for medical advice and early intervention.")
    else:
        st.success("✅ **Survival Chances Are Higher:** The model predicts a lower risk. Keep up healthy habits and regular check-ups!")

st.write("---")
st.caption(" This prediction is for educational purposes only. Always seek professional medical advice.")
