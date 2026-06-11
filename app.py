import streamlit as st
import pickle
import numpy as np
with open("model.pkl","rb") as file:
    model=pickle.load(file)

st.title("Student Performance Predictor")

hoursStudied=st.number_input("Hours Studied", min_value=0.0)
attendance=st.number_input("Attendance(%)", min_value=0.0, max_value=100.0)
previousMarks=st.number_input("Previous Marks", min_value=0.0)
sleepHours=st.number_input("Sleep Hours", min_value=0.0)

if st.button("Predict Score"):
    data=np.array([[hoursStudied,attendance,previousMarks,sleepHours]])
    prediction=model.predict(data)
    st.success(f"Predicted Score:{prediction[0]:.2f}")
