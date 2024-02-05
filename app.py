import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import plotly.graph_objects as go

# Custom CSS for layout, font, and background
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f5f5f5; /* Light gray background color */
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title for the app
st.title('Heart Health Analysis Dashboard')
st.markdown("A comprehensive tool for analyzing and predicting heart health risks.")

# Load the trained model
model_path = 'model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    
# Load heart data file
@st.cache(allow_output_mutation=True)
def load_data():
    # Since heart.csv is in the root directory of the project, reference it directly
    return pd.read_csv('heart.csv')

# Define the numerical and categorical features as per the model training
numerical_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# Define the mapping for categorical variables based on how the model was trained
cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
restecg_mapping = {'Normal': 0, 'Having ST-T Wave Abnormality': 1, 'Showing Probable or Definite Left Ventricular Hypertrophy': 2}
slp_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
thall_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

# Define a function to collect user inputs and create a DataFrame
def user_input_features():
    # Create a dictionary to store user inputs
    input_features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thall
    }

    # Create a DataFrame from the user inputs
    input_df = pd.DataFrame(input_features, index=[0])
    return input_df

data_heart = load_data()
# Define the input fields for the parameters
st.sidebar.header("User Input Parameters")

age = st.number_input('Age', min_value=18, max_value=120, value=30, step=1)
sex = st.selectbox('Sex', options=['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', options=cp_mapping.keys())
trtbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=90, max_value=200, value=120, step=1)
chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200, step=1)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', options=['Yes', 'No'])
restecg = st.selectbox('Resting Electrocardiographic Results', options=restecg_mapping.keys())
thalachh = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=100, step=1)
exng = st.radio('Exercise Induced Angina', options=['Yes', 'No'])
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slp = st.selectbox('The Slope of The Peak Exercise ST Segment', options=slp_mapping.keys())
caa = st.number_input('Number of Major Vessels (0-4) Colored by Fluoroscopy', min_value=0, max_value=4, value=0, step=1)
thall = st.selectbox('Thallium Stress Test Result', options=thall_mapping.keys())

# Convert categorical inputs to their corresponding numerical values
sex = 1 if sex == 'Male' else 0
cp = cp_mapping[cp]
fbs = 1 if fbs == 'Yes' else 0
restecg = restecg_mapping[restecg]

# When the user clicks the 'Predict' button
if st.button('Predict Risk'):
    # Create a DataFrame with the selected values
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thall
    }])

    
    processed_input = model.named_steps['preprocessor'].transform(input_df)  # If separate preprocessing step is needed
    probability = model.predict_proba(processed_input)[0][1]

    # Define a custom threshold
    custom_threshold = 0.3

    # Apply the custom threshold to determine the risk level
    if probability > custom_threshold:
        risk_level = "High Risk"
    elif probability > 0.2:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # Display the risk level
    st.success(f"The predicted risk level is: {risk_level} (Probability: {probability:.2f})")

# Sidebar with customization options
st.sidebar.header("Customization Options")
threshold = st.sidebar.slider("Risk Threshold", min_value=0.1, max_value=1.0, step=0.1, value=0.8)
high_risk_alert = st.sidebar.checkbox("High Risk Alerts", value=True)

# Predictive Analysis Section
st.subheader("Predictive Analysis")
st.markdown("Analyze patient data to predict heart disease risk.")
with st.expander("Filter Data"):
    age_range = st.slider('Select Age Range', min_value=int(data_heart['age'].min()), max_value=int(data_heart['age'].max()), value=[30, 60], key="age_range_slider")
    chol_range = st.slider('Select Cholesterol Range', min_value=int(data_heart['chol'].min()), max_value=int(data_heart['chol'].max()), value=[100, 200], key="chol_range_slider")
    filtered_data_heart = data_heart[(data_heart['age'] >= age_range[0]) & (data_heart['age'] <= age_range[1]) & (data_heart['chol'] >= chol_range[0]) & (data_heart['chol'] <= chol_range[1])]

st.dataframe(filtered_data_heart)

# Visualization Section
st.subheader("Data Visualization")
visualization_type = st.selectbox("Select Visualization Type", ["Scatter Plot", "Bar Chart", "Line Chart"], key="viz_type_selector")
if visualization_type == "Scatter Plot":
    fig = px.scatter(filtered_data_heart, x="chol", y="trtbps", color="output")
elif visualization_type == "Bar Chart":
    fig = px.bar(filtered_data_heart, x="cp", y="thalachh", color="output")
elif visualization_type == "Line Chart":
    fig = go.Figure(data=go.Scatter(x=filtered_data_heart['age'], y=filtered_data_heart['output'], mode='lines'))
st.plotly_chart(fig)

# Simulator for Preventive Scenarios
st.subheader("Preventive Scenario Simulator")
st.markdown("Adjust the parameters to simulate different preventive scenarios.")
sim_chol = st.slider("Cholesterol Level", int(data_heart["chol"].min()), int(data_heart["chol"].max()), int(data_heart["chol"].mean()))
sim_trtbps = st.slider("Resting Blood Pressure", int(data_heart["trtbps"].min()), int(data_heart["trtbps"].max()), int(data_heart["trtbps"].mean()))
sim_thalachh = st.slider("Maximum Heart Rate Achieved", int(data_heart["thalachh"].min()), int(data_heart["thalachh"].max()), int(data_heart["thalachh"].mean()))
sim_oldpeak = st.slider("ST Depression Induced by Exercise", float(data_heart["oldpeak"].min()), float(data_heart["oldpeak"].max()), float(data_heart["oldpeak"].mean()))

# Predict button
if st.button('Predict Heart Disease Risk') and filtered_data_heart is not None:
    # Create a DataFrame with the selected values
    input_data = pd.DataFrame([{
        'age': data_heart['age'].mean(),  # Mean age for the sake of example
        'trtbps': sim_trtbps,
        'chol': sim_chol,
        'thalachh': sim_thalachh,
        'oldpeak': sim_oldpeak,
        'sex': 1,  # 1 for male, 0 for female
        'cp': 0,  
        'fbs': 0,  # 0 for f
        'restecg': 0,  
        'exng': 0,  
        'slp': 0,  # Same as above
        'caa': 0,  # Same as above
        'thall': 0  # Same as above
    }])

    # Get user input and preprocess it
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thall
    }])

    # Predict the probability of the positive class
    probability = model.predict_proba(input_df)[0][1]

    # Define a custom threshold
    custom_threshold = 0.3

    # Apply the custom threshold to determine the risk level
    if probability > custom_threshold:
        risk_level = "High Risk"
    elif probability > 0.2:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # Display the risk level
    st.success(f"The predicted risk level is: {risk_level} (Probability: {probability:.2f})")

    # Use the model to predict
    prediction = model.predict(input_data)  # Assuming 'model' is your trained model
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    alert = "High Risk Alert" if risk_level == "High Risk" else "No Alert"

    # Display risk level and alert
    st.write(f"Risk Level: {risk_level}, Alert: {alert}")

