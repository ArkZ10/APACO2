import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="🔋 Solar Energy Predictor", layout="centered")
st.title("🔋 Solar Energy Production")
st.markdown("Daily energy summary and tomorrow's prediction")

# --- Load Model and Scalers ---
@st.cache_resource
def load_assets():
    model = load_model("solar_model.h5")  # Adjust path if needed
    with open("scaler_x.pkl", "rb") as fx:
        scaler_x = pickle.load(fx)
    with open("scaler_y.pkl", "rb") as fy:
        scaler_y = pickle.load(fy)
    return model, scaler_x, scaler_y

model, scaler_x, scaler_y = load_assets()

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_solar.csv", parse_dates=["Time"])
    return df

df_daily = load_data()

# --- Feature Selection ---
features = ['GHI_mean', 'temp_max', 'humidity_mean', 'wind_speed_mean']
target = 'Energy_theoretical_Wh_sum'

# --- Normalize Features and Target ---
X_scaled = scaler_x.transform(df_daily[features])
y_scaled = scaler_y.transform(df_daily[[target]])

# --- Create Sequences for Model Input ---
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# --- Today's Data ---
today_features = df_daily[features].iloc[-1]

# --- Interactive Inputs ---
st.subheader("🌤 Adjust Today's Weather (Optional)")

ghi = st.slider("GHI Mean (W/m²)", 0, 1000, int(today_features['GHI_mean']))
temp = st.slider("Max Temperature (°C)", -10, 50, int(today_features['temp_max']))
humidity = st.slider("Humidity Mean (%)", 0, 100, int(today_features['humidity_mean']))
wind_speed = st.slider(
    "Wind Speed Mean (m/s)",
    min_value=0.0,
    max_value=20.0,
    value=float(today_features['wind_speed_mean']),
    step=0.1
)

# --- Predict Button ---
if st.button("🔮 Predict Tomorrow's Energy"):
    # Create a new sequence with latest 23 points + today's edited input
    latest_sequence = X_scaled[-(time_steps-1):].tolist()  # Last 23 timesteps
    new_point = scaler_x.transform(np.array([[ghi, temp, humidity, wind_speed]]))[0]
    latest_sequence.append(new_point)
    latest_sequence = np.array(latest_sequence)

    # Reshape for model input
    latest_sequence = np.expand_dims(latest_sequence, axis=0)  # (1, time_steps, features)

    # Prediction
    pred_scaled = model.predict(latest_sequence)
    pred_energy = scaler_y.inverse_transform(pred_scaled)[0][0]

    # Show today's actual energy
    today_actual_energy = df_daily[target].iloc[-1]

    st.metric("Today's Energy Production", f"{today_actual_energy:.2f} Wh")
    st.metric("Tomorrow's Predicted Energy", f"{pred_energy:.2f} Wh")

# --- Footer ---
st.caption("Made with ❤️ using Streamlit")
