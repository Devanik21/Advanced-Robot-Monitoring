import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Advanced Robot Monitoring",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set random seed for reproducibility
np.random.seed(50)

# Number of records
n_records = 20000

# Generate random data for each feature
sound_level = np.random.normal(80, 30, n_records).astype(int)  # dcb, with some values reaching up to 150 dcb
temperature = np.random.normal(65, 15, n_records).astype(int)  # Celsius
battery_drain = np.random.choice([0, 1], size=n_records, p=[0.85, 0.15])  # 0 (normal) or 1 (draining fast)
process_halted = np.random.choice([0, 1], size=n_records, p=[0.95, 0.05])  # 0 (normal) or 1 (error halted)

# Define a more complex set of causes based on detailed conditions
cause = [
    "Oil Pressure Drop" if sl < 50 and temp < 50 and battery == 0 else
    "Overload Error" if sl > 130 and temp > 90 and halted == 1 else
    "Power Surge Alert" if battery == 1 and temp > 85 and 90 <= sl < 130 else
    "Vibration and Heat Spike" if 95 <= sl < 115 and temp > 80 else
    "Critical System Fault" if halted == 1 and temp > 90 and sl > 125 else
    "System Cooling Failure" if battery == 0 and temp >= 85 and sl < 90 else
    "Sensor Calibration Required" if (50 <= sl < 65 or 45 <= temp < 60) and battery == 1 else
    "Mechanical Fatigue Warning" if (sl > 110 and temp < 60) or (temp > 70 and halted == 1) else
    "Processor Overload" if battery == 1 and halted == 1 and temp < 75 else
    "Excessive Battery Drain" if battery == 1 and 60 <= temp < 80 else
    "Warning: High Temp Fluctuations" if np.abs(temp - np.mean(temperature)) > 20 else
    "Unstable Power Detected" if battery == 1 and sl < 70 and np.random.rand() < 0.05 else
    "Critical Shutdown Warning" if halted == 1 and sl > 130 else
    "Abnormal Sound Pattern" if sl >= 100 and sl <= 120 and np.random.rand() < 0.03 else
    "Sudden System Halt" if halted == 1 and sl < 70 else
    "Temperature Anomaly Detected" if temp > 75 and temp < 85 and np.random.rand() < 0.02 else
    "Frequent Battery Drain" if battery == 1 and np.random.rand() < 0.1 else
    "Energy Leakage" if battery == 1 and sl > 90 and np.random.rand() < 0.07 else
    "Unknown Error Code 37" if np.random.rand() < 0.01 else
    "Normal Operation"
    for sl, temp, battery, halted in zip(sound_level, temperature, battery_drain, process_halted)
]

# Create DataFrame
df = pd.DataFrame({
    "Sound_Level_dcb": sound_level,
    "Temperature_C": temperature,
    "Battery_Drain": battery_drain,
    "Process_Halted": process_halted,
    "Cause": cause
})
# Encode the categorical variable
label_encoder = LabelEncoder()
df['Cause_Encoded'] = label_encoder.fit_transform(df['Cause'])

# Split the data into features and target variable
X = df[['Sound_Level_dcb', 'Temperature_C', 'Battery_Drain', 'Process_Halted']]
y = df['Cause_Encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App Layout
st.title("Advanced Robot Monitoring Dashboard")

st.write(f"### Model Accuracy: {accuracy:.2f}")

st.image("2151329542.jpg")  
st.write("### Overview of Data")
st.write("This dashboard shows the monitoring data of an advanced robot, detailing sound levels, temperature, battery status, and any causes of alerts.")

# Display the DataFrame
st.write("#### Data Sample")
st.dataframe(df.sample(10))  # Display a random sample of 10 records

# Display unique causes
st.write("#### Unique Causes")
unique_causes = df['Cause'].unique()
st.write(unique_causes)

# Add a filter option
st.sidebar.title("Filter Options")
sound_filter = st.sidebar.slider("Sound Level (dcb)", min_value=df['Sound_Level_dcb'].min()-100, max_value=df['Sound_Level_dcb'].max()+100)
temperature_filter = st.sidebar.slider("Temperature (°C)", min_value=df['Temperature_C'].min()-50, max_value=df['Temperature_C'].max()+50)

# Filter the DataFrame based on sidebar inputs
filtered_df = df[(df['Sound_Level_dcb'] >= sound_filter) & (df['Temperature_C'] >= temperature_filter)]

# Display filtered results
st.write("#### Filtered Results")
st.dataframe(filtered_df)

# Prediction Section
st.sidebar.header("Predict Cause of Alert")
input_sound = st.sidebar.number_input("Input Sound Level (dcb)", min_value=0, max_value=150, value=80)
input_temp = st.sidebar.number_input("Input Temperature (°C)", min_value=0, max_value=150, value=65)
input_battery = st.sidebar.selectbox("Battery Drain", [0, 1])
input_halted = st.sidebar.selectbox("Process Halted", [0, 1])

# Predict cause based on user input
input_data = np.array([[input_sound, input_temp, input_battery, input_halted]])
predicted_cause_index = model.predict(input_data)[0]
predicted_cause = label_encoder.inverse_transform([predicted_cause_index])[0]

# Display prediction
st.write(f"#### Predicted Cause of Alert: **{predicted_cause}**")
