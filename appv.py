
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('laterite_model.h5')

# Create a scaler (same as the one used during training)
scaler = StandardScaler()

# Streamlit UI for input
st.title("Laterite Type Prediction")
st.write("Enter the features for prediction:")

# Collect user inputs
Ds = st.number_input('Ds', min_value=0.0)
UCS = st.number_input('UCS', min_value=0.0)
IS50 = st.number_input('IS50', min_value=0.0)
TS = st.number_input('TS', min_value=0.0)
Pw = st.number_input('Pw', min_value=0.0)
Di = st.number_input('Di', min_value=0.0)
Mc = st.number_input('Mc', min_value=0.0)
RQD = st.number_input('RQD', min_value=0.0)

# Prepare input features
input_features = np.array([[Ds, UCS, IS50, TS, Pw, Di, Mc, RQD]])

# Standardize input features (same transformation as during training)
input_features = scaler.fit_transform(input_features)

# Make predictions using the model
prediction = model.predict(input_features)
predicted_class = np.argmax(prediction, axis=1)

# Map the prediction to the corresponding Laterite Type
laterite_types = {0: "ILT", 1: "LT", 2: "LTC", 3: "LLT"}
predicted_laterite_type = laterite_types.get(predicted_class[0])

# Display prediction
st.write(f"The predicted Laterite Type is: {predicted_laterite_type}")
