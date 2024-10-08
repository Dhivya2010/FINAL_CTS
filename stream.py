import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the models
with open('best_rf.pkl', 'rb') as file:
    best_rf = pickle.load(file)

with open('best_model_fp_fsi.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Helper function to convert user input into model-compatible format
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

# Columns expected by the models
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

# Streamlit interface
st.title("Clinical Trial Readiness Prediction")

Country = st.selectbox("Country", ['Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'France', 'India', 'Italy', 'Japan', 'South Africa', 'Spain', 'UK', 'USA'])
development_unit = st.selectbox("Development Unit", ['Cardiovascular', 'NeuroScience', 'Oncology', 'Respiratory'])
phase = st.selectbox("Phase", ['Phase I', 'Phase II', 'Phase III', 'Phase IV'])
new_indication = st.selectbox("Is this a new indication?", ['Yes', 'No'])
Blinding = st.selectbox("Enter Blinding", ['Double Blind', 'Open Label', 'Single Blind'])
Pediatric = st.selectbox("Is this pediatric only?", ['Yes', 'No'])

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

input_array1 = convert_to_array(user_input_CTT_FP, columns1)
input_array2 = convert_to_array(user_input_FP_FSI, columns2)

input_2d_array1 = np.array(input_array1).reshape(1, -1)
input_2d_array2 = np.array(input_array2).reshape(1, -1)

if st.button('Predict'):
    try:
        y_pred1 = best_rf.predict(input_2d_array1)
        y_pred2 = best_model.predict(input_2d_array2)

        st.write("Predicted CTT-FP Weeks:", y_pred1[0], "Weeks")
        st.write("Predicted FP-FSI Weeks:", y_pred2[0], "Weeks")
        st.write("TOTAL WEEKS FOR CTT-FSI:", y_pred1[0] + y_pred2[0], "Weeks")

        total_weeks = y_pred1[0] + y_pred2[0]
        if total_weeks <= 105:
            st.subheader("High Demand")
            st.write("Resource Allocation:This site requires Allocation of additional personnel and equipment due to the high priority")
            st.write("Risks Identified:")
            st.write("- Potential delays due to rapid setup")
            st.write("Mitigation Strategies:")
            st.write("- Provide additional support and resources")
            st.write("- Implement expedited processes to align with the",y_pred1[0] + y_pred2[0],"weeks timeline")
        elif total_weeks <= 118:
            st.subheader("Medium Demand")
            st.write("Resource Allocation:This site requires Allocation of balanced amount of personnel and equipment. Ensure resources are sufficient but avoid over-allocation")
            st.write("Risks Identified:")
            st.write("-Risk of insufficient resources if allocation is not balanced.")
            st.write("-Pressure to meet the timeline can lead to scheduling conflicts or delays.")
            st.write("Mitigation Strategies:")
            st.write("-Distribute resources evenly to ensure all sites are adequately supported.")
            st.write("-Ensure that site readiness and setup processes are efficiently managed within the",y_pred1[0] + y_pred2[0]," weeks timeframe.")
        else:
            st.subheader("Low Demand")
            st.write("Resource Allocation:This site requires Minimal additional resources or only Maintenance of existing resource levels as the site is ahead of schedule.")
            st.write("Risks Identified:")
            st.write("-Extended preparation period may result in delays if not managed carefully.")
            st.write("-Potential for delays in resource availability due to phased allocation.")
            st.write("Mitigation Strategies:")
            st.write("-Continuously monitor progress and adjust plans as needed to prevent any emerging issues from causing delays.")
            st.write("-Implement long-term planning and monitoring to ensure readiness within the",y_pred1[0] + y_pred2[0],"weeks timeframe")



    except Exception as e:
        st.error(f"Error making predictions: {e}")
