import streamlit as st
import numpy as np
import joblib

# Load the best model and scaler
best_model = joblib.load('random_forest_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page title and icon
st.set_page_config(page_title='Fire Type Classifier', page_icon='🔥', layout='centered')

# App title and description
st.title('Fire Type Classifier')
st.markdown('This application predicts the type of fire based on MODIS satellite data.')

# Input features
brightness = st.number_input('Brightness', value=300.0)
bright_t31 = st.number_input('Brightness T31', value=290.0)
frp = st.number_input('Fire Radiative Power', value=100.0)
scan = st.number_input('Scan', value=0.5)
track = st.number_input('Track', value=0.5)
confidence = st.selectbox('Confidence', options=['High', 'Medium', 'Low'])

# Mapping confidence to numerical value
confidence_mapping = {'High': 90, 'Medium': 60, 'Low': 20}
confidence_value = confidence_mapping[confidence]

# Combine and scale input features
input_features = np.array([[brightness, bright_t31, frp, scan, track, confidence_value]])
scaled_features = scaler.transform(input_features)

# Make prediction and display result
if st.button('Predict Fire Type'):
    prediction = best_model.predict(scaled_features)[0]

    fire_types = {
        0: 'Presumed Vegetation Fire',
        1: 'Active Volcano',
        2: 'Other Static Land Source',
        3: 'Offshore',
    }

    st.success(f'The predicted fire type is: **{fire_types[prediction]}**')
