{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "2317331a-d182-4626-bfb5-294c66047f23",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "# Function to load the trained model\n",
    "@st.cache_resource\n",
    "def load_trained_model():\n",
    "    try:\n",
    "        model = load_model(\"multiclass_classifier.keras\")  # Replace with your model file\n",
    "        return model\n",
    "    except Exception as e:\n",
    "        st.error(\"Error loading the model. Ensure the 'multiclass_classifier.keras' file exists.\")\n",
    "        st.stop()  # Stop execution if the model is not loaded\n",
    "\n",
    "# Load the model\n",
    "model = load_trained_model()\n",
    "\n",
    "# Streamlit app title and description\n",
    "st.title(\"Laterite Type Classifier\")\n",
    "st.write(\"Enter the feature values in the sidebar to classify the type of laterite.\")\n",
    "\n",
    "# Sidebar for input fields\n",
    "st.sidebar.header(\"Input Features\")\n",
    "Ds = st.sidebar.number_input(\"Ds (mm)\", min_value=0.0, max_value=20.0, value=5.0, step=0.1)\n",
    "UCS = st.sidebar.number_input(\"UCS (MPa)\", min_value=0.0, max_value=25.0, value=10.0, step=0.1)\n",
    "IS50 = st.sidebar.number_input(\"IS50 (MPa)\", min_value=0.0, max_value=3.5, value=1.0, step=0.01)\n",
    "TS = st.sidebar.number_input(\"TS (MPa)\", min_value=0.0, max_value=3.0, value=1.5, step=0.01)\n",
    "Pw = st.sidebar.number_input(\"Pw (kg/m³)\", min_value=1400, max_value=3200, value=2000, step=10)\n",
    "Di = st.sidebar.number_input(\"Di (mm)\", min_value=1.5, max_value=3.5, value=2.0, step=0.01)\n",
    "Mc = st.sidebar.number_input(\"Mc (%)\", min_value=1.5, max_value=5.0, value=3.0, step=0.01)\n",
    "RQD = st.sidebar.number_input(\"RQD (%)\", min_value=10, max_value=50, value=30, step=1)\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"Classify\"):\n",
    "    try:\n",
    "        # Prepare input for prediction\n",
    "        input_features = np.array([[Ds, UCS, IS50, TS, Pw, Di, Mc, RQD]])\n",
    "        \n",
    "        # Make prediction\n",
    "        prediction = model.predict(input_features)\n",
    "        predicted_class = np.argmax(prediction)\n",
    "        \n",
    "        # Map class index to Laterite Type\n",
    "        laterite_types = [\"ILT\", \"LT\", \"LTC\", \"LLT\"]\n",
    "        result = laterite_types[predicted_class]\n",
    "        \n",
    "        # Display results\n",
    "        st.subheader(\"Classification Result\")\n",
    "        st.write(f\"The predicted Laterite Type is: **{result}**\")\n",
    "        st.write(f\"Confidence: **{prediction[0][predicted_class]:.2f}**\")\n",
    "    except Exception as e:\n",
    "        st.error(f\"An error occurred during prediction: {e}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
