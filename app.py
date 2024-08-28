import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved models
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Define the feature names
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                 'MonthlyCharges', 'TotalCharges']

def create_cohort(value):
    if value < 13:
        return '0-12 Months'
    elif value < 25:
        return '12-24 Months'
    elif value < 49:
        return '24-48 Months'
    else:
        return 'Over 48 Months'

st.title('Telco Customer Churn Prediction')

# Create input fields for each feature
input_data = {}

for feature in feature_names:
    if feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)
    elif feature == 'SeniorCitizen':
        input_data[feature] = st.selectbox(f"Select {feature}", [0, 1])
    else:
        unique_values = model.named_steps['preprocessor'].named_transformers_['categorical'].named_steps['ohe_category_df'].categories_[0]
        input_data[feature] = st.selectbox(f"Select {feature}", unique_values)

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Create the 'Tenure Cohort' feature
input_df['Tenure Cohort'] = input_df['tenure'].apply(create_cohort)

# Make sure all expected columns are present
expected_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'MonthlyCharges', 'TotalCharges', 'Tenure Cohort']

for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = None  # or some default value

# Make prediction
if st.button('Predict Churn'):
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        # Decode the prediction
        churn_status = le.inverse_transform(prediction)[0]
        churn_probability = probability[0][1]  # Probability of churning

        st.write(f"Churn Prediction: {churn_status}")
        st.write(f"Churn Probability: {churn_probability:.2f}")

        # Provide some interpretation
        if churn_status == 'Yes':
            st.write("This customer is likely to churn. Consider implementing retention strategies.")
        else:
            st.write("This customer is likely to stay. Continue providing good service.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check the input data and try again.")