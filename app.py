import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import plotly.express as px

# Streamlit configuration
st.set_page_config(page_title="Advanced Robot Monitoring", page_icon="🤖", layout="wide")

# Generate mock data
np.random.seed(50)
n_records = 20000
sound_level = np.random.normal(80, 30, n_records).astype(int)
temperature = np.random.normal(65, 15, n_records).astype(int)
battery_drain = np.random.choice([0, 1], size=n_records, p=[0.85, 0.15])
process_halted = np.random.choice([0, 1], size=n_records, p=[0.95, 0.05])

# Generate causes
def generate_cause(sl, temp, battery, halted):
    if sl < 50 and temp < 50 and battery == 0: return "Oil Pressure Drop"
    elif sl > 130 and temp > 90 and halted == 1: return "Overload Error"
    elif battery == 1 and temp > 85 and 90 <= sl < 130: return "Power Surge Alert"
    elif 95 <= sl < 115 and temp > 80: return "Vibration and Heat Spike"
    elif halted == 1 and temp > 90 and sl > 125: return "Critical System Fault"
    elif battery == 0 and temp >= 85 and sl < 90: return "System Cooling Failure"
    else: return "Normal Operation"

cause = [generate_cause(sl, temp, bat, hal) for sl, temp, bat, hal in zip(sound_level, temperature, battery_drain, process_halted)]

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

# Train model
X = df[['Sound_Level_dcb', 'Temperature_C', 'Battery_Drain', 'Process_Halted']]
y = df['Cause_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Layout
st.title("Advanced Robot Monitoring Dashboard")
st.write(f"### Model Accuracy: {accuracy:.2f}")
st.image("2151329542.jpg")  # Adjust the path to your image

# Data Visualization
st.write("### Data Visualizations")
fig = px.histogram(df, x='Sound_Level_dcb', color='Cause', title='Distribution of Sound Levels by Cause')
st.plotly_chart(fig)

# Display a Data Summary
st.write("### Data Summary")
st.write(df.describe())

# Filter Options
st.sidebar.title("Filter Options")
sound_filter = st.sidebar.slider("Sound Level (dcb)", min_value=int(df['Sound_Level_dcb'].min()), max_value=int(df['Sound_Level_dcb'].max()))
temperature_filter = st.sidebar.slider("Temperature (°C)", min_value=int(df['Temperature_C'].min()), max_value=int(df['Temperature_C'].max()))

# Filter the DataFrame based on sidebar inputs
filtered_df = df[(df['Sound_Level_dcb'] >= sound_filter) & (df['Temperature_C'] >= temperature_filter)]

# Display filtered results
st.write("### Filtered Results")
st.dataframe(filtered_df)

# Prediction Section
st.sidebar.header("Predict Cause of Alert")
input_sound = st.sidebar.number_input("Input Sound Level (dcb)", min_value=0, max_value=150, value=80)
input_temp = st.sidebar.number_input("Input Temperature (°C)", min_value=0, max_value=150, value=65)
input_battery = st.sidebar.selectbox("Battery Drain", [0, 1])
input_halted = st.sidebar.selectbox("Process Halted", [0, 1])

input_data = np.array([[input_sound, input_temp, input_battery, input_halted]])
predicted_cause_index = model.predict(input_data)[0]
predicted_cause = label_encoder.inverse_transform([predicted_cause_index])[0]

# Display prediction
st.write(f"### Predicted Cause of Alert: **{predicted_cause}**")

# Add download option
csv = filtered_df.to_csv(index=False)
st.sidebar.download_button("Download Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")
