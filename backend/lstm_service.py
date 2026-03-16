import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_and_predict_demand(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocessing
    # Convert timestamps safely
    df['connectionTime'] = pd.to_datetime(
        df['connectionTime'],
        format='mixed',
        errors='coerce'
    )

    df['disconnectTime'] = pd.to_datetime(
        df['disconnectTime'],
        format='mixed',
        errors='coerce'
    )

    # Remove invalid datetime rows
    df = df.dropna(subset=['connectionTime', 'disconnectTime'])

    # Feature extraction
    df['hour'] = df['connectionTime'].dt.hour
    df['day'] = df['connectionTime'].dt.day
    df['weekday'] = df['connectionTime'].dt.weekday

    # Charging duration (minutes)
    df['charging_duration'] = (
        df['disconnectTime'] - df['connectionTime']
    ).dt.total_seconds() / 60

    features = [
        'hour',
        'day',
        'weekday',
        'charging_duration'
    ]

    target = 'kWhDelivered'

    df = df[features + [target]].dropna()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_x.fit_transform(df[features])
    y = scaler_y.fit_transform(df[[target]])

    TIME_STEPS = 10

    X_seq, y_seq = create_sequences(X, y, TIME_STEPS)

    # If not enough data, return error or handle gracefully
    if len(X_seq) == 0:
        return {"error": "Not enough data to create sequences"}

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    # Model definition
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    # Train model
    # Reduced epochs for faster feedback during API calls, user can increase if needed
    history = model.fit(
        X_train, y_train,
        epochs=10, 
        batch_size=32,
        validation_split=0.1,
        verbose=0 
    )

    y_pred = model.predict(X_test)

    # Inverse scaling
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    
    return {
        "rmse": float(rmse),
        "message": "Model trained and evaluated successfully",
        "sample_prediction": float(y_pred_inv[-1][0]) if len(y_pred_inv) > 0 else None,
        "sample_actual": float(y_test_inv[-1][0]) if len(y_test_inv) > 0 else None
    }
