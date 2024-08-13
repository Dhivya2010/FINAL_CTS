import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your models (replace with your actual model file paths)
try:
    with open('best_model_fp_fsi.pkl', 'rb') as file:
        best_model = pickle.load(file)
    with open('best_rf.pkl', 'rb') as file2:
        best_rf = pickle.load(file2)
    st.write(f"Loaded model type: {type(best_rf)}")
except Exception as e:
    st.error(f"Error loading model: {e}")

def convert_to_array(user_input, columns):
    input_array = [0] * len(columns)
    for key, value in user_input.items():
        column_name = f"{key}_{value}"
        if column_name in columns:
            index = columns.get_loc(column_name)
            input_array[index] = 1
        else:
            st.warning(f"Column '{column_name}' not found in columns")
    return input_array

columns1 = pd.Index(['DevelopmentUnit_Cardiovascular', 'DevelopmentUnit_NeuroScience', 'DevelopmentUnit_Oncology', 'DevelopmentUnit_Respiratory',
                     'Phase_Phase I', 'Phase_Phase II', 'Phase_Phase III', 'Phase_Phase IV',
                     'New Indication_Yes', 'New Indication_No', 'Blinding_Double Blind', 'Blinding_Open Label', 'Blinding_Single Blind',
                     'Pediatric_Yes', 'Pediatric_No'])

columns2 = pd.Index(['Country_Argentina', 'Country_Australia', 'Country_Brazil', 'Country_Canada', 'Country_China', 'Country_France',
                     'Country_India', 'Country_Italy', 'Country_Japan', 'Country_South Africa', 'Country_Spain', 'Country_UK', 'Country_USA',
                     'DevelopmentUnit_Cardiovascular', 'DevelopmentUnit_NeuroScience', 'DevelopmentUnit_Oncology', 'DevelopmentUnit_Respiratory',
                     'Phase_Phase I', 'Phase_Phase II', 'Phase_Phase III', 'Phase_Phase IV',
                     'New Indication_Yes', 'New Indication_No', 'Blinding_Double Blind', 'Blinding_Open Label', 'Blinding_Single Blind',
                     'Pediatric_Yes', 'Pediatric_No'])

# Streamlit inputs
Country = st.selectbox("Country", ['Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'France', 'India', 'Italy', 'Japan', 'South Africa', 'Spain', 'UK', 'USA'])
development_unit = st.selectbox("Development Unit", ['Cardiovascular', 'NeuroScience', 'Oncology', 'Respiratory'])
phase = st.selectbox("Phase", ['Phase I', 'Phase II', 'Phase III', 'Phase IV'])
new_indication = st.selectbox("Is this a new indication?", ['Yes', 'No'])
Blinding = st.selectbox("Enter Blinding", ['Double Blind', 'Open Label', 'Single Blind'])
Pediatric = st.selectbox("Is this pediatric only?", ['Yes', 'No'])
# user input store
user_input = {
    'Country': Country,
    'DevelopmentUnit': development_unit,
    'Phase': phase,
    'New Indication': new_indication,
    'Blinding': Blinding,
    'Pediatric': Pediatric
}

user_input_CTT_FP = {
    'DevelopmentUnit': user_input['DevelopmentUnit'],
    'Phase': user_input['Phase'],
    'New Indication': user_input['New Indication'],
    'Blinding': user_input['Blinding'],
    'Pediatric': user_input['Pediatric']
}

user_input_FP_FSI = {
    'Country': user_input['Country'],
    'DevelopmentUnit': user_input['DevelopmentUnit'],
    'Phase': user_input['Phase'],
    'New Indication': user_input['New Indication'],
    'Blinding': user_input['Blinding'],
    'Pediatric': user_input['Pediatric']
}

# Converting the user input to arrays
input_array1 = convert_to_array(user_input_CTT_FP, columns1)
input_array2 = convert_to_array(user_input_FP_FSI, columns2)

# Converting the 1D arrays to 2D arrays
input_2d_array1 = np.array(input_array1).reshape(1, -1)
input_2d_array2 = np.array(input_array2).reshape(1, -1)

if st.button('Predict'):
    try:
        if hasattr(best_rf, 'predict') and hasattr(best_model, 'predict'):
            y_pred1 = best_rf.predict(input_2d_array1)
            y_pred2 = best_model.predict(input_2d_array2)

            st.write("Predicted CTT-FP Weeks:", y_pred1[0], "Weeks")
            st.write("Predicted FP-FSI Weeks:", y_pred2[0], "Weeks")
            st.write("TOTAL WEEKS FOR CTT-FSI:", y_pred1[0] + y_pred2[0], "Weeks")
        else:
            st.error("Loaded objects are not models with a predict method.")
    except Exception as e:
        st.error(f"Error making predictions: {e}")
