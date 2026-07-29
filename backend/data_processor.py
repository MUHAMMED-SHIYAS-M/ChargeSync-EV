import pandas as pd
import json
from datetime import datetime, timezone
import os


def parse_user_inputs(inputs):
    try:
        if pd.isna(inputs) or inputs == "":
            return []
        if isinstance(inputs, str):
            return json.loads(inputs)
        return inputs
    except Exception as e:
        print(f"Error parsing userInputs: {e}")
        return []


def enrich_data(df):
    """Adds derived features. Works with both raw session data and aggregated hourly data."""
    # --- Raw session format (connectionTime / disconnectTime columns) ---
    if 'connectionTime' in df.columns and 'disconnectTime' in df.columns:
        for col in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        if 'doneChargingTime' in df.columns and 'disconnectTime' in df.columns:
            df['doneChargingTime'] = df['doneChargingTime'].fillna(df['disconnectTime'])
        df = df.dropna(subset=['connectionTime', 'disconnectTime'])
        df['duration_stay'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600
        df['duration_charging'] = (df['doneChargingTime'] - df['connectionTime']).dt.total_seconds() / 3600
        if 'kWhDelivered' in df.columns:
            charging_hrs = df['duration_charging'].clip(lower=0.01)
            df['avg_power'] = df['kWhDelivered'] / charging_hrs
        return df

    # --- Aggregated hourly format (acn_dataset.csv) ---
    if 'hour' in df.columns or 'session_count' in df.columns:
        df = df.reset_index(drop=True)
        base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        df['connectionTime'] = base + pd.to_timedelta(df.index, unit='h')
        df['disconnectTime'] = df['connectionTime'] + pd.Timedelta(hours=1)
        df['doneChargingTime'] = df['disconnectTime']
        df['duration_stay'] = 1.0
        df['duration_charging'] = 1.0

        if 'avg_kWh' in df.columns:
            df['kWhDelivered'] = df['avg_kWh']
            df['avg_power'] = df['avg_kWh']

        # Synthesize a stationID column from hour_of_day buckets
        if 'stationID' not in df.columns:
            if 'hour_of_day' in df.columns:
                df['stationID'] = df['hour_of_day'].apply(lambda h: f"SYN-{int(h) % 10 + 1:02d}")
            else:
                df['stationID'] = "SYN-01"

        if 'siteID' not in df.columns:
            df['siteID'] = "ACN-SITE"
        if 'userID' not in df.columns:
            df['userID'] = "SYN-USER"

        return df

    return df


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    if 'userInputs' in df.columns:
        df['userInputs'] = df['userInputs'].apply(parse_user_inputs)
    return enrich_data(df)


def get_global_stats(df):
    """Calculates aggregate metrics for the dashboard."""
    kwh_col = 'kWhDelivered' if 'kWhDelivered' in df.columns else 'total_kWh' if 'total_kWh' in df.columns else None
    total_kwh = float(df[kwh_col].sum()) if kwh_col else 0.0

    avg_wait = float(df['duration_stay'].mean() * 60) if 'duration_stay' in df.columns else 0.0

    if 'connectionTime' in df.columns:
        peak_count = int(df.groupby(pd.Grouper(key='connectionTime', freq='h')).size().max())
    elif 'session_count' in df.columns:
        peak_count = int(df['session_count'].max())
    else:
        peak_count = 0

    return {
        "totalNodes": int(df['stationID'].nunique()) if 'stationID' in df.columns else 0,
        "avgWaitMinutes": round(avg_wait, 1),
        "totalMegawatts": round(total_kwh / 1000, 2),
        "peakLoad": peak_count
    }


def get_station_occupancy(df, timestamp):
    """Returns number of active sessions at a given timestamp."""
    if 'connectionTime' not in df.columns or 'disconnectTime' not in df.columns:
        return {}
    active = df[(df['connectionTime'] <= timestamp) & (df['disconnectTime'] > timestamp)]
    if 'stationID' in active.columns:
        return active.groupby('stationID').size().to_dict()
    return {}


if __name__ == "__main__":
    DATA_PATH = r"d:\games\group11ziipp\group11\backend\acn_dataset.csv"
    if os.path.exists(DATA_PATH):
        processed_df = load_and_preprocess(DATA_PATH)
        print(f"Processed {len(processed_df)} rows.")
        print(processed_df.head())
    else:
        print("Data file not found.")
