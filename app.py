import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
@st.cache_resource
def load_model(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
    return None

model = load_model("model.joblib")

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction")

if not model:
    st.error("⚠️ Model not found. Please train the model first.")
else:
    with st.form("prediction_form"):
        st.subheader("Enter Employee Details")

        experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
        education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
        role = st.selectbox("Job Role", ["Engineer", "Manager", "HR", "Analyst", "Developer"])
        department = st.selectbox("Department", ["IT", "Finance", "HR", "Operations", "Marketing"])
        location = st.selectbox("Location", ["New York", "San Francisco", "London", "Berlin", "Remote"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        input_df = pd.DataFrame([{
            "Experience": experience,
            "Education": education,
            "Role": role,
            "Department": department,
            "Location": location,
            "Gender": gender
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
