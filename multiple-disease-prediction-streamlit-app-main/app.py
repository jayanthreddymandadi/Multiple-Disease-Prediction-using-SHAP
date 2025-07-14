import os
import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

working_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Sidebar for disease selection
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML And SHAP')
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
    user_input = [st.text_input(feature) for feature in feature_names]

    if st.button('Diabetes Test Result'):
        try:
            user_input = np.array([float(x) for x in user_input]).reshape(1, -1)
        except ValueError:
            st.error("Please enter valid numbers for all fields.")
        else:
            prediction = diabetes_model.predict(user_input)
            st.success('The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic')

            if prediction[0] == 1:  # SHAP Visualization only for positive results
                explainer = shap.KernelExplainer(diabetes_model.predict, np.zeros((1, user_input.shape[1])))
                shap_values = explainer.shap_values(user_input)
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                                                      feature_names=feature_names), show=False)
                plt.close(fig)
                st.pyplot(fig)

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML And SHAP')
    feature_names = ['Age', 'Sex', 'Chest Pain types', 'Resting Blood Pressure', 'Serum Cholestoral',
                     'Fasting Blood Sugar > 120 mg/dl', 'Resting ECG results', 'Max Heart Rate achieved',
                     'Exercise Induced Angina', 'Oldpeak', 'Slope of Peak Exercise ST',
                     'Number of Major Vessels', 'Thalassemia']
    
    # Input fields
    user_input = []
    for feature in feature_names:
        if feature == "Sex":
            user_input.append(st.selectbox("Sex", ["Male", "Female", "Other"]))  # Restricted options
        else:
            user_input.append(st.text_input(feature))

    if st.button('Heart Disease Test Result'):
        try:
            # Convert numerical inputs
            user_input_values = [float(x) if feature != "Sex" else (1 if x == "Male" else 0) for x, feature in zip(user_input, feature_names)]
            user_input_array = np.array(user_input_values).reshape(1, -1)
        except ValueError:
            st.error("Please enter valid numbers for all fields.")
        else:
            prediction = heart_disease_model.predict(user_input_array)

            if prediction[0] == 1:
                st.success('**The person has heart disease!**')

                explainer = shap.KernelExplainer(heart_disease_model.predict, np.zeros((1, user_input_array.shape[1])))
                shap_values = explainer.shap_values(user_input_array)
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                                                      feature_names=feature_names), show=False)
                plt.close(fig)
                st.pyplot(fig)
            else:
                st.error('**The person does NOT have heart disease**')


# Parkinson's Prediction (Optimized SHAP Visualization)
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML And SHAP")

    feature_names = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
                     "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                     "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1",
                     "spread2", "D2", "PPE"]

    user_input = [st.text_input(feature, key=f"parkinson_{feature}") for feature in feature_names]

    if st.button("Parkinson's Test Result"):
        try:
            user_array = np.array([float(x) for x in user_input]).reshape(1, -1)
        except ValueError:
            st.error("Please enter valid numeric values for all features.")
        else:
            prediction = parkinsons_model.predict(user_array)

            if prediction[0] == 1:
                st.success("The person has Parkinson's disease")

                # Load or simulate lightweight background dataset
                try:
                    background = pd.read_csv(f"{working_dir}/saved_models/parkinsons_background.csv").sample(20)
                except FileNotFoundError:
                    st.warning("Background dataset not found — using synthetic sample for SHAP.")
                    background = np.random.normal(0, 1, size=(20, len(feature_names)))

                # SHAP explanation
                explainer = shap.KernelExplainer(parkinsons_model.predict, background)
                shap_values = explainer.shap_values(user_array)

                # Visualize all features in waterfall plot
                st.subheader("Feature Importance (Waterfall Plot)")
                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=user_array[0],
                    feature_names=feature_names
                )

                fig, ax = plt.subplots(figsize=(7, 5))
                shap.plots.waterfall(explanation, show=False)
                plt.tight_layout()
                plt.close(fig)
                st.pyplot(fig)

            else:
                st.error("The person does not have Parkinson's disease")