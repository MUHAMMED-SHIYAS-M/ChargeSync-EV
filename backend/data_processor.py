import pandas as pd
import json
from datetime import datetime, timezone
import os


def parse_user_inputs(inputs):
    try:
        if pd.isna(inputs) or inputs == "":
            return []
        # JSON string is often wrapped in extra quotes or escaped
        if isinstance(inputs, str):
            return json.loads(inputs)
        return inputs
    except Exception as e:
        print(f"Error parsing userInputs: {e}")
        return []

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Convert times to datetime with UTC and error coercion
    df['connectionTime'] = pd.to_datetime(df['connectionTime'], utc=True, errors='coerce')
    df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], utc=True, errors='coerce')
    
    # If doneChargingTime is missing, use disconnectTime as fallback
    df['doneChargingTime'] = pd.to_datetime(df['doneChargingTime'], utc=True, errors='coerce')
    df['doneChargingTime'] = df['doneChargingTime'].fillna(df['disconnectTime'])
    
    # Drop rows where critical times are missing
    df = df.dropna(subset=['connectionTime', 'disconnectTime'])
    
    # Parse nested userInputs
    df['userInputs'] = df['userInputs'].apply(parse_user_inputs)
    
    # Feature Engineering
    df['duration_stay'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600
    df['duration_charging'] = (df['doneChargingTime'] - df['connectionTime']).dt.total_seconds() / 3600
    
    # Calculate energy efficiency or power
    # Use clip to avoid negative or zero duration
    charging_hrs = df['duration_charging'].clip(lower=0.01)
    df['avg_power'] = df['kWhDelivered'] / charging_hrs
    
    return df

def get_global_stats(df):
    """Calculates aggregate metrics for the dashboard."""
    total_kwh = df['kWhDelivered'].sum()
    avg_wait = df['duration_stay'].mean() * 60 # Convert to minutes as a proxy for wait
    active_now = len(df[df['disconnectTime'] > pd.Timestamp.now(tz='UTC')]) # This might be empty for historical data
    
    # For historical data demonstration, let's take a peak period
    peak_count = df.groupby(pd.Grouper(key='connectionTime', freq='h')).size().max()
    
    return {
        "totalNodes": df['stationID'].nunique(),
        "avgWaitMinutes": round(avg_wait, 1),
        "totalMegawatts": round(total_kwh / 1000, 2),
        "peakLoad": int(peak_count)
    }

def get_station_occupancy(df, timestamp):
    """Returns number of active sessions at a given timestamp."""
    active = df[(df['connectionTime'] <= timestamp) & (df['disconnectTime'] > timestamp)]
    return active.groupby('stationID').size().to_dict()

if __name__ == "__main__":
    DATA_PATH = r"c:\Users\naizan\Desktop\projects\group11\acn-data (1) (1).csv"
    if os.path.exists(DATA_PATH):
        processed_df = load_and_preprocess(DATA_PATH)
        print(f"Processed {len(processed_df)} sessions.")
        print(processed_df.head())
    else:
        print("Data file not found.")
