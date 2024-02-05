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

# Title for your app
st.title('Heart Health Analysis Dashboard')
st.markdown("A comprehensive tool for analyzing and predicting heart health risks.")

# Load your trained model
model_path = 'model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    
@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv('heart.csv')

data_heart = load_data()

# Debugging: Print or display the first few rows of the DataFrame to check its structure
st.write("First few rows of data:", data_heart.head())

# Define the numerical and categorical features as per the model training
numerical_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# Map categorical features to their respective options (based on your training dataset)
sex_options = [0, 1]  # Assuming 0 for female, 1 for male
cp_options = [0, 1, 2, 3]  # As per the type of chest pain
fbs_options = [0, 1]  # Assuming 0 for FBS <= 120 mg/dl, 1 for FBS > 120 mg/dl
restecg_options = [0, 1, 2]  # Resting electrocardiographic results options
exng_options = [0, 1]  # Exercise induced angina options
slp_options = [0, 1, 2]  # Slope of the peak exercise ST segment options
caa_options = [0, 1, 2, 3, 4]  # Number of major vessels colored by fluoroscopy
thall_options = [0, 1, 2, 3]  # Thallium stress test results

# Define the input fields for the parameters
st.sidebar.header("User Input Parameters")

# Function to collect user inputs
def user_input_features():
    # Numerical inputs
    age = st.sidebar.number_input('Age', min_value=18, max_value=120, value=30, step=1)
    trtbps = st.sidebar.number_input('Resting Blood Pressure (in mm Hg)', min_value=90, max_value=200, value=120, step=1)
    chol = st.sidebar.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200, step=1)
    thalachh = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=100, step=1)
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.0, value=1.0, step=0.1)

    # Categorical inputs
    sex = st.sidebar.selectbox('Sex', options=sex_options)
    cp = st.sidebar.selectbox('Chest Pain Type', options=cp_options)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options=fbs_options)
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', options=restecg_options)
    exng = st.sidebar.selectbox('Exercise Induced Angina', options=exng_options)
    slp = st.sidebar.selectbox('The Slope of The Peak Exercise ST Segment', options=slp_options)
    caa = st.sidebar.selectbox('Number of Major Vessels (0-4) Colored by Fluoroscopy', options=caa_options)
    thall = st.sidebar.selectbox('Thallium Stress Test Result', options=thall_options)

    # Create a DataFrame from the user inputs
    input_data = {
        'age': age, 'trtbps': trtbps, 'chol': chol, 'thalachh': thalachh, 'oldpeak': oldpeak,
        'sex': sex, 'cp': cp, 'fbs': fbs, 'restecg': restecg, 'exng': exng, 'slp': slp, 'caa': caa, 'thall': thall
    }
    features = pd.DataFrame(input_data, index=[0])
    return features

# Display the user input features
input_df = user_input_features()

# Predict button
if st.button('Predict Risk'):
    # Use the model to make predictions
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.write(f'Prediction: {prediction[0]}')
    st.write(f'Prediction Probability: {probability:.2f}')

    # Define a custom threshold
    custom_threshold = 0.6  # This is an example, adjust based on your needs

    # Apply the custom threshold to determine the risk level
    if probability > custom_threshold:
        risk_level = "High Risk"
    elif probability > 0.4:  # You can define another cutoff for moderate risk
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
if st.button('Predict Heart Disease Risk'):
    # Create a DataFrame from the user inputs
    input_df = pd.DataFrame([{
        'age': age,
        'trtbps': trtbps,
        'chol': chol,
        'thalachh': thalachh,
        'oldpeak': oldpeak,
        'sex': sex,
        'cp': cp,
        'fbs': fbs,
        'restecg': restecg,
        'exng': exng,
        'slp': slp,
        'caa': caa,
        'thall': thall
    }])

    # Predict the probability of the positive class (e.g., high risk)
    probability = model.predict_proba(input_df)[0][1]

    # Define a custom threshold
    custom_threshold = 0.6  # This is an example, adjust based on your needs

    # Apply the custom threshold to determine the risk level
    if probability > custom_threshold:
       risk_level = "High Risk"
    elif probability > 0.4:  # You can define another cutoff for moderate risk
       risk_level = "Moderate Risk"
    else:
     risk_level = "Low Risk"

    # Display the risk level
    st.success(f"The predicted risk level is: {risk_level} (Probability: {probability:.2f})")

    # Use your model to predict
    prediction = model.predict(input_df)  # Use input_df for prediction
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    alert = "High Risk Alert" if risk_level == "High Risk" else "No Alert"

    # Display risk level and alert
    st.write(f"Risk Level: {risk_level}, Alert: {alert}")
