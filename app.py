import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Function to load the trained model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("multiclass_classifier.keras")  # Replace with your model file
        return model
    except Exception as e:
        st.error("Error loading the model. Ensure the 'laterite_classifier.h5' file exists.")
        st.stop()  # Stop execution if the model is not loaded

model = load_trained_model()

# Title and description
st.title("Laterite Type Classifier")
st.write("Enter the feature values to classify the type of laterite.")

# Input fields for features
st.sidebar.header("Input Features")
Ds = st.sidebar.number_input("Ds (mm)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
UCS = st.sidebar.number_input("UCS (MPa)", min_value=0.0, max_value=25.0, value=10.0, step=0.1)
IS50 = st.sidebar.number_input("IS50 (MPa)", min_value=0.0, max_value=3.5, value=1.0, step=0.01)
TS = st.sidebar.number_input("TS (MPa)", min_value=0.0, max_value=3.0, value=1.5, step=0.01)
Pw = st.sidebar.number_input("Pw (kg/m³)", min_value=1400, max_value=3200, value=2000, step=10)
Di = st.sidebar.number_input("Di (mm)", min_value=1.5, max_value=3.5, value=2.0, step=0.01)
Mc = st.sidebar.number_input("Mc (%)", min_value=1.5, max_value=5.0, value=3.0, step=0.01)
RQD = st.sidebar.number_input("RQD (%)", min_value=10, max_value=50, value=30, step=1)

# Predict button
if st.button("Classify"):
    try:
        # Prepare input for prediction
        input_features = np.array([[Ds, UCS, IS50, TS, Pw, Di, Mc, RQD]])
        
        # Make prediction
        prediction = model.predict(input_features)
        predicted_class = np.argmax(prediction)
        
        # Map class index to Laterite Type
        laterite_types = ["ILT", "LT", "LTC", "LLT"]
        result = laterite_types[predicted_class]
        
        # Display results
        st.subheader("Classification Result")
        st.write(f"The predicted Laterite Type is: **{result}**")
        st.write(f"Confidence: **{prediction[0][predicted_class]:.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
