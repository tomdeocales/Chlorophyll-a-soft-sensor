import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import json
import re
import pytz
from datetime import datetime
import threading


# Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyAwZRqZ3nkYJ-2zcm3g7sUBL1mIUNuPO1A",
  "authDomain": "tadlac-299f8.firebaseapp.com",
  "databaseURL": "https://tadlac-299f8-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "tadlac-299f8",
  "storageBucket": "tadlac-299f8.firebasestorage.app",
  "messagingSenderId": "568323038212",
  "appId": "1:568323038212:web:4918f3da2ce4ee1c4ea951"
}
# Firebase Realtime Database URL
database_url = firebase_config['databaseURL']

# Load the stacking model and scaler
stacking_model_loaded = joblib.load('stacking_ensemble_model.joblib')
scaler_loaded = joblib.load('scaler.pkl')

# Streamlit app title
st.title('Chlorophyll Prediction and Forecasting Model')

# Function to sanitize Firebase keys
def sanitize_key(key):
    """Sanitize Firebase keys to remove invalid characters."""
    return re.sub(r'[$#\[\]/.]', '_', key)

# Recursive function to sanitize nested dictionaries
def sanitize_dict(d):
    """Recursively sanitize keys in a nested dictionary."""
    if isinstance(d, dict):
        return {sanitize_key(k): sanitize_dict(v) for k, v in d.items()}
    else:
        return d

# Function to fetch the latest sensor data from Firebase using the REST API
def get_latest_sensor_data():
    try:
        sensor_data_ref = f'{database_url}/SensorData.json'
        response = requests.get(sensor_data_ref)
        
        if response.status_code == 200:
            sensor_data = response.json()
            # Sort the data by timestamp and get the latest entry
            latest_data = max(sensor_data.values(), key=lambda x: x['timestamp'])
            
            # Sanitize the data and extract timestamp
            sanitized_data = sanitize_dict(latest_data)
            timestamp = datetime.fromtimestamp(sanitized_data['timestamp'] / 1000.0)
            singapore_tz = pytz.timezone('Asia/Singapore')
            timestamp = timestamp.replace(tzinfo=pytz.utc).astimezone(singapore_tz)

            # Prepare the sensor data as a DataFrame
            sensor_df = pd.DataFrame({
                'Temp (°C)': [sanitized_data['Temperature']],
                'Turbidity (FNU)': [sanitized_data['Turbidity']],
                'pH': [sanitized_data['pH']],
                'DO (mg/L)': [sanitized_data['DissolvedOxygen']],
                'year': [timestamp.year],
                'month': [timestamp.month],
                'day': [timestamp.day],
                'day_of_week': [timestamp.weekday()],
                'day_of_year': [timestamp.timetuple().tm_yday],
                'quarter': [(timestamp.month - 1) // 3 + 1],
                'hour': [timestamp.hour]
            })
            return sensor_df, timestamp
        else:
            st.write(f"Error fetching data from Firebase: {response.text}")
            return pd.DataFrame(), None
    except Exception as e:
        st.write(f"Error fetching data from Firebase: {e}")
        return pd.DataFrame(), None

# Forecast future chlorophyll levels based on the latest data
def forecast_future_chlorophyll(df, stacking_model, scaler, features, steps=10, uncertainty_factor=0.1):
    last_row = df[features].iloc[-1:].copy()

    forecasts = []
    upper_bounds = []
    lower_bounds = []

    for i in range(steps):
        # Simulate a "next" time step (roll over the day/month)
        last_row['day_of_year'] += 1
        last_row['day'] = (last_row['day'] % 31) + 1
        last_row['day_of_week'] = (last_row['day_of_week'] + 1) % 7
        if last_row['day'].iloc[0] == 1:
            last_row['month'] = (last_row['month'] % 12) + 1
            if last_row['month'].iloc[0] == 1:
                last_row['year'] += 1
        last_row['quarter'] = (last_row['month'] - 1) // 3 + 1

        # Scaling and prediction
        last_row_scaled = scaler.transform(last_row)
        forecast_log = stacking_model.predict(last_row_scaled)
        forecast = np.expm1(forecast_log)  # Reverse log transformation

        # Calculate the upper and lower bounds
        upper_bound = forecast[0] * (1 + uncertainty_factor)  # Add 10% to forecast
        lower_bound = forecast[0] * (1 - uncertainty_factor)  # Subtract 10% from forecast

        forecasts.append(forecast[0])
        upper_bounds.append(upper_bound)
        lower_bounds.append(lower_bound)

    return forecasts, upper_bounds, lower_bounds

# Save predictions and forecasts to Firebase using the REST API
def save_prediction_to_firebase(predicted_chlorophyll, sensor_data, forecast_values, upper_bounds, lower_bounds):
    predicted_chlorophyll = float(predicted_chlorophyll)
    
    # Use only the first value from forecast, upper bound, and lower bound arrays
    first_forecast_value = float(forecast_values[0])
    first_upper_bound = float(upper_bounds[0])
    first_lower_bound = float(lower_bounds[0])

    # Sanitize sensor data keys
    sensor_data_values = {
        sanitize_key(key): float(value[0]) if isinstance(value[0], np.float64) else int(value[0])
        for key, value in sensor_data.items()
    }

    # Push the prediction and forecast data to Firebase
    prediction_ref = f'{database_url}/Predictions.json'
    singapore_tz = pytz.timezone('Asia/Singapore')
    timestamp = datetime.now(singapore_tz).strftime('%Y-%m-%d %H:%M:%S')
    data = {
        **sensor_data_values,
        'Predicted_Chlorophyll': predicted_chlorophyll,
        'Forecasted_Chlorophyll': first_forecast_value,
        'Upper_Bound_Chlorophyll': first_upper_bound,
        'Lower_Bound_Chlorophyll': first_lower_bound,
        'timestamp': timestamp
    }
    response = requests.post(prediction_ref, json=data)

    if response.status_code != 200:
        st.write(f"Error saving prediction: {response.text}")

    # Update the current prediction separately
    current_prediction_ref = f'{database_url}/CurrentPrediction.json'
    current_data = {
        'Predicted_Chlorophyll': predicted_chlorophyll,
        'timestamp': timestamp
    }
    # Make a PUT request to update the current prediction
    current_response = requests.put(current_prediction_ref, json=current_data)

    if current_response.status_code != 200:
        st.write(f"Error updating current prediction: {current_response.text}")

# Background task to periodically fetch data, make predictions, and save to Firebase
def background_task():
    if 'last_processed_timestamp' not in st.session_state:
        st.session_state.last_processed_timestamp = None

    while True:
        # Fetch the latest sensor data
        fetched_sensor_data, timestamp = get_latest_sensor_data()

        if not fetched_sensor_data.empty:
            # Check if the timestamp has changed (to avoid processing the same data)
            if timestamp != st.session_state.last_processed_timestamp:
                st.session_state.last_processed_timestamp = timestamp

                # Proceed with prediction and saving logic
                features = ['Temp (°C)', 'Turbidity (FNU)', 'pH', 'DO (mg/L)', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 'quarter', 'hour']
                X_new_scaled = scaler_loaded.transform(fetched_sensor_data[features])
                y_new_pred_log = stacking_model_loaded.predict(X_new_scaled)
                y_new_pred = np.expm1(y_new_pred_log)

                future_forecasts, upper_bounds, lower_bounds = forecast_future_chlorophyll(fetched_sensor_data, stacking_model_loaded, scaler_loaded, features, steps=10)
                save_prediction_to_firebase(y_new_pred[0], fetched_sensor_data, future_forecasts, upper_bounds, lower_bounds)
        else:
            st.write("No new sensor data.")
        
        time.sleep(30)  # Wait 60 seconds before fetching new data



# Start background task in a separate thread
if __name__ == '__main__':
    # Start the background task in a separate thread
    threading.Thread(target=background_task, daemon=True).start()

    # Main Streamlit app code (will run continuously while the background task works)
    st.write("## Continuous Chlorophyll Prediction and Forecasting")
    st.write("Prediction tasks are running in the background, check Firebase for updated predictions.")
