import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Set the page title and description
st.title("UCLA Admission Predictor")
st.write("""
This app predicts whether a student is likely to be admitted to UCLA 
based on various academic and research factors.
""")

# Load the trained Neural Network model
with open("models/Neuralmodel.pkl", "rb") as model_file:
    nn_model = pickle.load(model_file)

with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# User Input Form
with st.form("student_input_form"):
    st.subheader("Student Profile")
    
    GRE_Score = st.number_input("GRE Score", min_value=260, max_value=340, step=1)
    TOEFL_Score = st.number_input("TOEFL Score", min_value=80, max_value=120, step=1)
    University_Rating = st.selectbox("University Rating", options=[1, 2, 3, 4, 5])
    SOP = st.slider("Statement of Purpose (SOP) Strength", min_value=1.0, max_value=5.0, step=0.5)
    LOR = st.slider("Letter of Recommendation (LOR) Strength", min_value=1.0, max_value=5.0, step=0.5)
    CGPA = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.01)
    Research = st.selectbox("Research Experience", options=["Yes", "No"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Admission Chance")

# If form is submitted
if submitted:
    # Convert categorical input
    Research_0 = 1 if Research == "No" else 0
    Research_1 = 1 if Research == "Yes" else 0

    # One-hot encode University_Rating
    University_Rating_1 = 1 if University_Rating == 1 else 0
    University_Rating_2 = 1 if University_Rating == 2 else 0
    University_Rating_3 = 1 if University_Rating == 3 else 0
    University_Rating_4 = 1 if University_Rating == 4 else 0
    University_Rating_5 = 1 if University_Rating == 5 else 0
    
    # Prepare input data in the correct order
    input_data = [[GRE_Score, TOEFL_Score, SOP, LOR, CGPA,
                            University_Rating_1, University_Rating_2, University_Rating_3,
                            University_Rating_4, University_Rating_5,
                            Research_0, Research_1]]
    
    input_data_scaled = scaler.transform(input_data)

    # Predict admission outcome (1 = Admitted, 0 = Not Admitted)
    prediction = nn_model.predict(input_data_scaled)


    # Display result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("Congratulations! You have a high chance of being admitted to UCLA. 🎓")
    else:
        st.error("Unfortunately, your admission chances are low. Consider improving your profile. 📚")

st.write(
    """We used a Neural Network model to predict your admission chances. 
    Plotting loss curve below."""
)
st.image("loss_curve.png")
