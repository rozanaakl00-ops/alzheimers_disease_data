# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaling_numeric_cols.pkl", "rb"))

# Columns used during training (in correct order)
feature_cols = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
                'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
                'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
                'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
                'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
                'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
                'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
                'Forgetfulness']

numeric_cols = ['Age','BMI','AlcoholConsumption','PhysicalActivity','DietQuality',
                'SleepQuality','SystolicBP','DiastolicBP','CholesterolTotal',
                'CholesterolLDL','CholesterolHDL','CholesterolTriglycerides',
                'MMSE','FunctionalAssessment','ADL']

# Create Streamlit inputs
st.title("Alzheimer's Disease Prediction")
st.write("Enter patient information:")

input_data = {}
for col in feature_cols:
    if col in numeric_cols:
        input_data[col] = st.number_input(col, value=0.0)
    else:  # categorical (assume 0/1)
        input_data[col] = st.selectbox(col, options=[0,1])

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale numeric columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Ensure column order matches training
input_df = input_df[feature_cols]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"Predicted Diagnosis: **{prediction}**")
    st.write(f"Probability of Alzheimer’s: **{probability*100:.2f}%**")
