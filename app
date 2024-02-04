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
# Load your trained model
model_path = 'model.pkl'  # Updated to relative path
model = None
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Load heart data file
def load_data():
    # Use a relative path from the Streamlit app script to the CSV file
    return pd.read_csv('data/heart.csv')

data_heart = load_data()

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
        'sex': 1,  # Assuming 1 for male, 0 for female, adjust as necessary
        'cp': 0,  # You need to adjust this based on your encoding
        'fbs': 0,  # Assuming 0 for false, adjust as necessary
        'restecg': 0,  # Adjust according to your model's training data
        'exng': 0,  # Same as above
        'slp': 0,  # Same as above
        'caa': 0,  # Same as above
        'thall': 0  # Same as above
        # Add other necessary features with default or mean values
    }])

    # Use your model to predict
    prediction = model.predict(input_data)  # Assuming 'model' is your trained model
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    alert = "High Risk Alert" if risk_level == "High Risk" else "No Alert"

    # Display risk level and alert
    st.write(f"Risk Level: {risk_level}, Alert: {alert}")
