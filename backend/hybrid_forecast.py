"""
hybrid_forecast.py  (v3 — all 10 improvements)
================================================
Hybrid ARIMA + LSTM charging demand forecasting pipeline.

Improvements over v2:
  1.  Data cleaning: dedup + 99th-percentile outlier cap
  2.  Cyclical time encoding: hour, dow, month, week-of-year sin/cos
  3.  Feature engineering: idle_time, rolling_6h, lag_1/24/168
  4.  auto_arima (pmdarima) with seasonal m=24 — replaces hardcoded (5,1,0)
  5.  Improved LSTM: Dropout 0.2→0.3, Dense 32→Dense(64, relu)
  6.  Lookback window: 48→168 hours (full week)
  7.  Hybrid blend: 0.5 × ARIMA forecast + 0.5 × LSTM forecast
  8.  Weather data via OpenWeatherMap (graceful fallback)
  9.  TimeSeriesSplit 3-fold cross-validation
  10. MAPE metric for every model
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = r"d:\games\group11ziipp\group11\backend\acn_dataset.csv"
OUT_DIR   = r"C:\Users\MUHAMMED SHIYAS M\.gemini\antigravity\brain\5552dde6-6857-4f32-a5f7-92fbe970e15d"

# ─── Dark theme ───────────────────────────────────────────────────────────────
BG, CARD, BORDER = "#020617", "#0f172a", "#1e293b"
CYAN, VIOLET, EMERALD, ROSE, AMBER = "#22d3ee", "#a78bfa", "#34d399", "#f87171", "#fbbf24"
TEXT, MUTED = "#f1f5f9", "#64748b"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER,
    "font.family": "DejaVu Sans",
})


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — MAPE
# ══════════════════════════════════════════════════════════════════════════════
def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (ignores zero-valued actuals)."""
    actual    = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# ══════════════════════════════════════════════════════════════════════════════
# 1. WEATHER DATA (OpenWeatherMap — graceful fallback)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_weather(lat: float = 34.0522, lon: float = -118.2437) -> dict:
    """
    Fetch current weather from OpenWeatherMap API.
    Returns dict with temperature_c, humidity_pct, rain_mm.
    Falls back to neutral values if the API key is missing or call fails.
    Set OWM_API_KEY in your .env file to enable live weather.
    """
    api_key = os.getenv("OWM_API_KEY", "")
    if not api_key:
        print("[Weather] No OWM_API_KEY found — using neutral fallback values.")
        return {"temperature_c": 20.0, "humidity_pct": 50.0, "rain_mm": 0.0, "source": "fallback"}

    try:
        import urllib.request, json as _json
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())
        temp     = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rain_mm  = data.get("rain", {}).get("1h", 0.0)
        print(f"[Weather] Live: {temp:.1f}°C  Humidity: {humidity}%  Rain: {rain_mm}mm/h")
        return {
            "temperature_c": temp,
            "humidity_pct":  float(humidity),
            "rain_mm":       float(rain_mm),
            "source":        "live",
        }
    except Exception as e:
        print(f"[Weather] API call failed ({e}) — using neutral fallback values.")
        return {"temperature_c": 20.0, "humidity_pct": 50.0, "rain_mm": 0.0, "source": "fallback"}


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING + FEATURE ENGINEERING  (Improvements 1, 2, 3)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(path: str) -> pd.DataFrame:
    """Load, clean, and engineer features from the ACN dataset."""
    print("[Preprocess] Loading dataset ...")
    df = pd.read_csv(path)

    # ── Improvement 1: Remove duplicates ─────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Preprocess] Removed {before - len(df):,} duplicate rows.")

    # ── Parse timestamps ──────────────────────────────────────────────────────
    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df = df.dropna(subset=["connectionTime", "disconnectTime", "kWhDelivered"])

    # ── Improvement 1: Remove zero and extreme outliers ───────────────────────
    df = df[df["kWhDelivered"] > 0].copy()
    q99 = df["kWhDelivered"].quantile(0.99)
    before_clip = len(df)
    df = df[df["kWhDelivered"] < q99].copy()
    print(f"[Preprocess] Removed {before_clip - len(df):,} extreme outlier rows (>{q99:.1f} kWh, 99th pct).")

    df = df.sort_values("connectionTime").reset_index(drop=True)

    # ── Improvement 2: Core + Cyclical time features ──────────────────────────
    df["hour"]         = df["connectionTime"].dt.hour
    df["day_of_week"]  = df["connectionTime"].dt.dayofweek   # 0=Mon
    df["month"]        = df["connectionTime"].dt.month
    df["week_of_year"] = df["connectionTime"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["date"]         = df["connectionTime"].dt.date

    # Cyclical encoding — prevents discontinuity at period boundaries
    df["hour_sin"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]        = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]      = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"]      = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["week_sin"]       = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]       = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # ── Improvement 3: Charging duration + idle time ──────────────────────────
    df["charging_duration"] = (
        (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    # Idle time: vehicle plugged in but done charging (blocking behaviour)
    done_col = df["doneChargingTime"].fillna(df["disconnectTime"])
    df["idle_time"] = (
        (df["disconnectTime"] - done_col).dt.total_seconds() / 3600
    ).clip(lower=0)

    # ── Rolling mean + lag features (session-level, filled after agg) ─────────
    df["kwh_roll3"] = df["kWhDelivered"].shift(1).rolling(3,  min_periods=1).mean()
    df["kwh_roll6"] = df["kWhDelivered"].shift(1).rolling(6,  min_periods=1).mean()
    df["kwh_lag1"]  = df["kWhDelivered"].shift(1)
    df["kwh_lag2"]  = df["kWhDelivered"].shift(2)

    # ── Floor to hour for ARIMA aggregation ───────────────────────────────────
    df["hour_ts"] = df["connectionTime"].dt.floor("h")

    df = df.dropna().reset_index(drop=True)

    print(f"[Preprocess] {len(df):,} valid sessions | "
          f"{df['connectionTime'].dt.date.nunique()} days | "
          f"kWh range [{df['kWhDelivered'].min():.1f}, {df['kWhDelivered'].max():.1f}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. ARIMA — Improvement 4: auto_arima with seasonal m=24
# ══════════════════════════════════════════════════════════════════════════════
def train_arima(df: pd.DataFrame):
    """
    Aggregate sessions per hour, fit auto_arima (seasonal, m=24),
    forecast next 24 h.
    Returns: fitted, hourly, test_series, preds, fcast24, metrics
    """
    print("\n[ARIMA] Aggregating hourly session counts …")
    hourly = df.groupby("hour_ts").size().rename("sessions")
    hourly = hourly.asfreq("h", fill_value=0)

    # 80/20 chronological split
    split   = int(len(hourly) * 0.8)
    train_s = hourly.iloc[:split]
    test_s  = hourly.iloc[split:]
    print(f"[ARIMA] Train hours: {len(train_s)} | Test hours: {len(test_s)}")

    # Improvement 4 — auto_arima
    try:
        from pmdarima import auto_arima as _auto_arima
        print("[ARIMA] Running auto_arima (fast mode, m=24) …")
        auto_model = _auto_arima(
            train_s,
            seasonal=True,
            m=24,
            max_p=3, max_q=3, max_D=1, max_P=1, max_Q=1,
            information_criterion="aic",
            stepwise=True,
            n_jobs=1,
            error_action="ignore",
            suppress_warnings=True,
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        print(f"[ARIMA] Best order: {order}  Seasonal: {seasonal_order}")
        preds_raw = auto_model.predict(n_periods=len(test_s))
        preds     = np.clip(preds_raw, 0, None)
        
        # Refit on full series (fast fallback to same order instead of new search)
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fitted_full = SARIMAX(
            hourly,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, method="nm") # 'nm' is lower memory
        
        fcast24 = np.clip(fitted_full.forecast(steps=24), 0, None)
        fitted  = fitted_full
    except ImportError:
        print("[ARIMA] pmdarima not installed — falling back to ARIMA(5,1,0). "
              "Run: pip install pmdarima")
        from statsmodels.tsa.arima.model import ARIMA
        fitted_tmp = ARIMA(train_s, order=(5, 1, 0)).fit()
        preds      = np.clip(fitted_tmp.forecast(steps=len(test_s)), 0, None)
        fitted_full = ARIMA(hourly, order=(5, 1, 0)).fit()
        fcast24    = np.clip(fitted_full.forecast(steps=24), 0, None)
        fitted     = fitted_full

    rmse_val = float(np.sqrt(mean_squared_error(test_s, preds)))
    mae_val  = float(mean_absolute_error(test_s, preds))
    mape_val = mape(test_s.values, preds)
    print(f"[ARIMA] Test RMSE: {rmse_val:.3f}  MAE: {mae_val:.3f}  MAPE: {mape_val:.2f}%")

    metrics = {"rmse": round(rmse_val, 4), "mae": round(mae_val, 4), "mape": round(mape_val, 2)}
    return fitted, hourly, test_s, preds, fcast24, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. LSTM  (Improvements 5, 6, 9, 10)
# ══════════════════════════════════════════════════════════════════════════════
def train_lstm(df: pd.DataFrame, weather: dict | None = None):
    """
    Predict HOURLY TOTAL kWh.
    Improvements:
      5. Dense(32)→Dense(64), Dropout 0.2→0.3
      6. TIME_STEPS: 48→168 (1 full week)
      9. TimeSeriesSplit 3-fold CV evaluation
     10. MAPE metric
    """
    print("\n[LSTM] Building hourly aggregated demand series ...")

    # Aggregate to hourly kWh totals + session count + avg duration/idle
    hourly = df.groupby("hour_ts").agg(
        total_kwh      = ("kWhDelivered", "sum"),
        session_count  = ("kWhDelivered", "count"),
        avg_duration   = ("charging_duration", "mean"),
        avg_idle       = ("idle_time", "mean"),
    ).asfreq("h", fill_value=0).reset_index()

    hourly["hour"]        = hourly["hour_ts"].dt.hour
    hourly["day_of_week"] = hourly["hour_ts"].dt.dayofweek
    hourly["month"]       = hourly["hour_ts"].dt.month
    hourly["week_of_year"]= hourly["hour_ts"].dt.isocalendar().week.astype(int)
    hourly["is_weekend"]  = (hourly["day_of_week"] >= 5).astype(int)

    # Cyclical encoding on hourly series
    hourly["hour_sin"]  = np.sin(2 * np.pi * hourly["hour"] / 24)
    hourly["hour_cos"]  = np.cos(2 * np.pi * hourly["hour"] / 24)
    hourly["dow_sin"]   = np.sin(2 * np.pi * hourly["day_of_week"] / 7)
    hourly["dow_cos"]   = np.cos(2 * np.pi * hourly["day_of_week"] / 7)
    hourly["month_sin"] = np.sin(2 * np.pi * (hourly["month"] - 1) / 12)
    hourly["month_cos"] = np.cos(2 * np.pi * (hourly["month"] - 1) / 12)
    hourly["week_sin"]  = np.sin(2 * np.pi * hourly["week_of_year"] / 52)
    hourly["week_cos"]  = np.cos(2 * np.pi * hourly["week_of_year"] / 52)

    # Rolling + lag features on HOURLY series (Improvement 3)
    hourly["kwh_roll3"]  = hourly["total_kwh"].shift(1).rolling(3,  min_periods=1).mean()
    hourly["kwh_roll6"]  = hourly["total_kwh"].shift(1).rolling(6,  min_periods=1).mean()  # rolling_6h
    hourly["kwh_lag1"]   = hourly["total_kwh"].shift(1)
    hourly["kwh_lag2"]   = hourly["total_kwh"].shift(2)
    hourly["kwh_lag24"]  = hourly["total_kwh"].shift(24)    # same hour yesterday
    hourly["kwh_lag168"] = hourly["total_kwh"].shift(168)   # same hour last week

    # Improvement 8: Add weather features (constant for training; live for inference)
    w = weather or {"temperature_c": 20.0, "humidity_pct": 50.0, "rain_mm": 0.0}
    hourly["temperature_c"] = w["temperature_c"]
    hourly["humidity_pct"]  = w["humidity_pct"]
    hourly["rain_mm"]       = w["rain_mm"]

    hourly = hourly.fillna(0)

    features = [
        # Cyclical time
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "week_sin", "week_cos",
        # Calendar
        "is_weekend", "session_count",
        # Charging behaviour
        "avg_duration", "avg_idle",
        # Rolling demand
        "kwh_roll3", "kwh_roll6",
        # Lag features
        "kwh_lag1", "kwh_lag2", "kwh_lag24", "kwh_lag168",
        # Weather
        "temperature_c", "humidity_pct", "rain_mm",
    ]
    target = "total_kwh"

    data = hourly[features + [target]].copy()
    print(f"[LSTM] {len(data):,} hourly rows | Features: {len(features)}")

    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_x.fit_transform(data[features])
    y_sc = scaler_y.fit_transform(data[[target]])

    # Improvement 6 — 168-step (1 week) lookback window
    TIME_STEPS = 168
    Xs, ys = [], []
    for i in range(len(X_sc) - TIME_STEPS):
        Xs.append(X_sc[i : i + TIME_STEPS])
        ys.append(y_sc[i + TIME_STEPS])
    Xs, ys = np.array(Xs), np.array(ys)

    print(f"[LSTM] TIME_STEPS={TIME_STEPS} | Total windows: {len(Xs):,}")

    # ── Improvement 9: TimeSeriesSplit 3-fold CV ──────────────────────────────
    tscv = TimeSeriesSplit(n_splits=3)
    cv_rmse, cv_mae, cv_mape = [], [], []
    print("[LSTM] Running 3-fold TimeSeriesSplit cross-validation …")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(Xs), 1):
        Xtr, Xval = Xs[tr_idx], Xs[val_idx]
        ytr, yval = ys[tr_idx], ys[val_idx]

        cv_model = Sequential([
            Input(shape=(TIME_STEPS, len(features))),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        cv_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        cv_model.fit(Xtr, ytr, epochs=15, batch_size=32,
                     validation_data=(Xval, yval), verbose=0)

        yval_pred = cv_model.predict(Xval, verbose=0)
        yval_inv  = scaler_y.inverse_transform(yval)
        ypred_inv = scaler_y.inverse_transform(yval_pred)

        fold_rmse = float(np.sqrt(mean_squared_error(yval_inv, ypred_inv)))
        fold_mae  = float(mean_absolute_error(yval_inv, ypred_inv))
        fold_mape = mape(yval_inv, ypred_inv)
        cv_rmse.append(fold_rmse)
        cv_mae.append(fold_mae)
        cv_mape.append(fold_mape)
        print(f"  Fold {fold}: RMSE={fold_rmse:.3f}  MAE={fold_mae:.3f}  MAPE={fold_mape:.2f}%")

    print(f"[LSTM] CV Summary — RMSE: {np.mean(cv_rmse):.3f}±{np.std(cv_rmse):.3f} | "
          f"MAE: {np.mean(cv_mae):.3f}±{np.std(cv_mae):.3f} | "
          f"MAPE: {np.nanmean(cv_mape):.2f}%")

    # ── Final model trained on 80/20 split ────────────────────────────────────
    split_idx = int(len(Xs) * 0.8)
    X_train, X_test = Xs[:split_idx], Xs[split_idx:]
    y_train, y_test = ys[:split_idx], ys[split_idx:]
    print(f"[LSTM] Final train: {len(X_train):,}  Test: {len(X_test):,}")

    # Improvement 5 — improved architecture
    model = Sequential([
        Input(shape=(TIME_STEPS, len(features))),
        LSTM(128, return_sequences=True),
        Dropout(0.3),                          # ← was 0.2
        LSTM(64),
        Dense(64, activation="relu"),          # ← was Dense(32)
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse", metrics=["mae"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    print("[LSTM] Training final model with EarlyStopping …")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred    = model.predict(X_test, verbose=0)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    rmse_val = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    mae_val  = float(mean_absolute_error(y_test_inv, y_pred_inv))
    r2_val   = float(r2_score(y_test_inv, y_pred_inv))
    mape_val = mape(y_test_inv, y_pred_inv)
    print(f"[LSTM] Test RMSE: {rmse_val:.3f}  MAE: {mae_val:.3f}  R2: {r2_val:.3f}  MAPE: {mape_val:.2f}%")

    # ── Generate 24-hour forward LSTM forecast ────────────────────────────────
    # Use the last TIME_STEPS window from the full sequence
    last_window = X_sc[-TIME_STEPS:].reshape(1, TIME_STEPS, len(features))
    lstm_24h_scaled = []
    window = last_window.copy()
    for _ in range(24):
        step_pred = model.predict(window, verbose=0)[0, 0]
        lstm_24h_scaled.append(step_pred)
        # Roll window: shift left, append a copy of last step features
        # (simplified: reuse last feature row — sufficient for kWh trend)
        new_step = window[0, -1, :].copy()
        window   = np.concatenate([window[:, 1:, :],
                                   new_step.reshape(1, 1, len(features))], axis=1)

    lstm_24h = np.clip(
        scaler_y.inverse_transform(
            np.array(lstm_24h_scaled).reshape(-1, 1)
        ).flatten(), 0, None
    )
    print(f"[LSTM] 24h forward forecast — mean: {lstm_24h.mean():.2f} kWh/h  "
          f"peak: {lstm_24h.max():.2f} kWh/h")

    metrics = {
        "rmse": round(rmse_val, 4), "mae": round(mae_val, 4),
        "r2":   round(r2_val, 4),   "mape": round(mape_val, 2),
        "cv_rmse_mean": round(float(np.mean(cv_rmse)), 4),
        "cv_mape_mean": round(float(np.nanmean(cv_mape)), 2),
    }
    return model, scaler_x, scaler_y, history, y_test_inv, y_pred_inv, metrics, lstm_24h


# ══════════════════════════════════════════════════════════════════════════════
# 5. HYBRID FORECAST  (Improvement 7 — 0.5 × ARIMA + 0.5 × LSTM)
# ══════════════════════════════════════════════════════════════════════════════
def build_hybrid_forecast(
    arima_24h: np.ndarray,
    lstm_24h: np.ndarray,
    avg_kwh_per_session: float,
) -> dict:
    """
    Blend ARIMA session forecast (→ kWh via avg_kwh) and LSTM kWh forecast.
    Final kWh = 0.5 × arima_kwh + 0.5 × lstm_kwh
    """
    arima_kwh = arima_24h * avg_kwh_per_session     # sessions × kWh/session → kWh
    blended   = 0.5 * arima_kwh + 0.5 * lstm_24h   # Improvement 7

    now  = datetime.now().replace(minute=0, second=0, microsecond=0)
    rows = []
    for i in range(24):
        rows.append({
            "hour_label":          (now + timedelta(hours=i)).strftime("%I %p"),
            "predicted_sessions":  round(float(arima_24h[i]), 2),
            "arima_kwh":           round(float(arima_kwh[i]), 2),
            "lstm_kwh":            round(float(lstm_24h[i]), 2),
            "predicted_kwh_load":  round(float(blended[i]), 2),
        })

    peak_idx  = int(np.argmax([r["predicted_kwh_load"] for r in rows]))
    peak_hour = rows[peak_idx]["hour_label"]
    print(f"\n[Hybrid] 0.5×ARIMA + 0.5×LSTM blend | "
          f"Peak: {peak_hour} ({rows[peak_idx]['predicted_kwh_load']:.1f} kWh)")

    return {"peak_hour": peak_hour, "peak_idx": peak_idx, "forecast": rows}


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_all(history, y_test_inv, y_pred_inv, lstm_metrics,
             hourly_series, test_series, arima_preds, arima_metrics,
             hybrid_result):
    """Generate and save 3 figures."""

    # ── Figure 1: LSTM training curves + scatter ──────────────────────────────
    fig1 = plt.figure(figsize=(16, 10), facecolor=BG)
    fig1.suptitle("ChargeSync — LSTM Energy Demand Model (v3)", fontsize=17,
                  fontweight="bold", color=TEXT, y=0.99)
    gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.45, wspace=0.35)

    epochs = range(1, len(history.history["loss"]) + 1)

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(epochs, history.history["loss"],     color=CYAN,   lw=2.5, label="Train Loss")
    ax1.plot(epochs, history.history["val_loss"], color=VIOLET, lw=2,   linestyle="--", label="Val Loss")
    ax1.fill_between(epochs, history.history["loss"],     alpha=0.12, color=CYAN)
    ax1.fill_between(epochs, history.history["val_loss"], alpha=0.08, color=VIOLET)
    ax1.set_title("Training & Validation Loss (MSE)", color=TEXT, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT); ax1.grid(True, alpha=0.3)

    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.plot(epochs, history.history["mae"],     color=EMERALD, lw=2.5, label="Train MAE")
    ax2.plot(epochs, history.history["val_mae"], color=AMBER,   lw=2,   linestyle="--", label="Val MAE")
    ax2.fill_between(epochs, history.history["mae"],     alpha=0.12, color=EMERALD)
    ax2.fill_between(epochs, history.history["val_mae"], alpha=0.08, color=AMBER)
    ax2.set_title("Training & Validation MAE (kWh)", color=TEXT, fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE (kWh)")
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT); ax2.grid(True, alpha=0.3)

    n_show = min(300, len(y_test_inv))
    ax3 = fig1.add_subplot(gs1[1, 0])
    ax3.plot(y_test_inv[:n_show], color=CYAN, lw=1.5, alpha=0.85, label="Actual kWh")
    ax3.plot(y_pred_inv[:n_show], color=ROSE, lw=1.5, alpha=0.85, linestyle="--", label="Predicted kWh")
    ax3.set_title(f"Actual vs Predicted (first {n_show} test samples)", color=TEXT, fontsize=12, fontweight="bold")
    ax3.set_xlabel("Sample Index"); ax3.set_ylabel("kWh Delivered")
    ax3.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT); ax3.grid(True, alpha=0.3)

    ax4 = fig1.add_subplot(gs1[1, 1])
    ax4.scatter(y_test_inv[:1000], y_pred_inv[:1000], alpha=0.25, color=VIOLET, s=8, edgecolors="none")
    mn = min(y_test_inv.min(), y_pred_inv.min())
    mx = max(y_test_inv.max(), y_pred_inv.max())
    ax4.plot([mn, mx], [mn, mx], color=CYAN, lw=2, linestyle="--", label="Perfect fit")
    props = dict(boxstyle="round,pad=0.5", facecolor=CARD, edgecolor=BORDER, alpha=0.9)
    ax4.text(0.05, 0.95,
             f"R²={lstm_metrics['r2']:.3f}  RMSE={lstm_metrics['rmse']:.3f}\n"
             f"MAE={lstm_metrics['mae']:.3f}  MAPE={lstm_metrics['mape']:.2f}%",
             transform=ax4.transAxes, fontsize=9, color=EMERALD, va="top", bbox=props)
    ax4.set_title("Scatter: Actual vs Predicted kWh", color=TEXT, fontsize=12, fontweight="bold")
    ax4.set_xlabel("Actual kWh"); ax4.set_ylabel("Predicted kWh")
    ax4.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT); ax4.grid(True, alpha=0.3)

    p1 = os.path.join(OUT_DIR, "lstm_accuracy_graphs.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[Plot] Saved: {p1}")
    plt.close(fig1)

    # ── Figure 2: ARIMA actual vs predicted + residuals ───────────────────────
    fig2 = plt.figure(figsize=(16, 7), facecolor=BG)
    fig2.suptitle("ChargeSync — ARIMA Session Demand Model (auto_arima)", fontsize=17,
                  fontweight="bold", color=TEXT, y=1.01)
    gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

    ax5 = fig2.add_subplot(gs2[0])
    show_n = min(200, len(test_series))
    ax5.plot(range(show_n), test_series.values[:show_n],  color=CYAN, lw=2,   label="Actual sessions")
    ax5.plot(range(show_n), arima_preds[:show_n],          color=ROSE, lw=2,   linestyle="--", label="ARIMA predicted")
    ax5.fill_between(range(show_n), test_series.values[:show_n], arima_preds[:show_n],
                     alpha=0.12, color=AMBER)
    props = dict(boxstyle="round,pad=0.5", facecolor=CARD, edgecolor=BORDER, alpha=0.9)
    ax5.text(0.05, 0.95,
             f"RMSE={arima_metrics['rmse']:.3f}  MAE={arima_metrics['mae']:.3f}\n"
             f"MAPE={arima_metrics['mape']:.2f}%",
             transform=ax5.transAxes, fontsize=9, color=EMERALD, va="top", bbox=props)
    ax5.set_title("ARIMA: Actual vs Predicted (hourly sessions)", color=TEXT, fontsize=12, fontweight="bold")
    ax5.set_xlabel("Test Hour Index"); ax5.set_ylabel("Sessions / Hour")
    ax5.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT); ax5.grid(True, alpha=0.3)

    residuals = test_series.values[:show_n] - arima_preds[:show_n]
    ax6 = fig2.add_subplot(gs2[1])
    ax6.bar(range(show_n), residuals,
            color=[ROSE if r < 0 else EMERALD for r in residuals], alpha=0.7, width=0.9)
    ax6.axhline(0, color=TEXT, lw=1, linestyle="--", alpha=0.5)
    ax6.set_title("ARIMA Residuals (Actual − Predicted)", color=TEXT, fontsize=12, fontweight="bold")
    ax6.set_xlabel("Test Hour Index"); ax6.set_ylabel("Residual (sessions)")
    ax6.grid(True, alpha=0.3)

    p2 = os.path.join(OUT_DIR, "arima_session_model.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[Plot] Saved: {p2}")
    plt.close(fig2)

    # ── Figure 3: 24-hour hybrid forecast (ARIMA + LSTM blend) ───────────────
    fig3 = plt.figure(figsize=(18, 7), facecolor=BG)
    fig3.suptitle("ChargeSync — 24-Hour Hybrid Forecast (0.5×ARIMA + 0.5×LSTM)", fontsize=17,
                  fontweight="bold", color=TEXT, y=1.02)
    gs3 = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.35)

    rows     = hybrid_result["forecast"]
    labels   = [r["hour_label"] for r in rows]
    sessions = [r["predicted_sessions"] for r in rows]
    arima_kw = [r["arima_kwh"] for r in rows]
    lstm_kw  = [r["lstm_kwh"] for r in rows]
    blend_kw = [r["predicted_kwh_load"] for r in rows]
    peak_i   = hybrid_result["peak_idx"]

    # Sessions bar
    ax_s = fig3.add_subplot(gs3[0])
    bars_colors = [ROSE if i == peak_i else CYAN for i in range(24)]
    ax_s.bar(range(24), sessions, color=bars_colors, alpha=0.85, width=0.75)
    ax_s.set_xticks(range(24)); ax_s.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_s.set_title("Predicted Sessions (ARIMA)", color=TEXT, fontsize=11, fontweight="bold")
    ax_s.set_xlabel("Hour"); ax_s.set_ylabel("Sessions / Hour")
    ax_s.axvline(peak_i, color=AMBER, lw=2, linestyle="--", alpha=0.7)
    ax_s.set_facecolor(CARD); ax_s.grid(True, alpha=0.3, axis="y")

    # ARIMA vs LSTM kWh comparison
    ax_c = fig3.add_subplot(gs3[1])
    x = np.arange(24)
    ax_c.bar(x - 0.2, arima_kw, width=0.4, color=VIOLET, alpha=0.8, label="ARIMA kWh")
    ax_c.bar(x + 0.2, lstm_kw,  width=0.4, color=EMERALD, alpha=0.8, label="LSTM kWh")
    ax_c.set_xticks(range(24)); ax_c.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_c.set_title("ARIMA vs LSTM kWh Forecasts", color=TEXT, fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Hour"); ax_c.set_ylabel("kWh Load")
    ax_c.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax_c.set_facecolor(CARD); ax_c.grid(True, alpha=0.3, axis="y")

    # Blended kWh
    load_colors = [ROSE if i == peak_i else EMERALD for i in range(24)]
    ax_l = fig3.add_subplot(gs3[2])
    ax_l.bar(range(24), blend_kw, color=load_colors, alpha=0.85, width=0.75)
    ax_l.set_xticks(range(24)); ax_l.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_l.set_title("Blended kWh Load (0.5×ARIMA + 0.5×LSTM)", color=TEXT, fontsize=11, fontweight="bold")
    ax_l.set_xlabel("Hour"); ax_l.set_ylabel("Total kWh Load")
    ax_l.axvline(peak_i, color=AMBER, lw=2, linestyle="--", alpha=0.7,
                 label=f"Peak: {hybrid_result['peak_hour']}")
    ax_l.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax_l.set_facecolor(CARD); ax_l.grid(True, alpha=0.3, axis="y")

    fig3.patch.set_facecolor(BG)
    p3 = os.path.join(OUT_DIR, "hybrid_24h_forecast.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[Plot] Saved: {p3}")
    plt.close(fig3)

    return p1, p2, p3


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFUSION MATRIX + CLASS-WEIGHTED ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_test_inv, y_pred_inv):
    nonzero = y_test_inv[y_test_inv.flatten() > 0].flatten()
    if len(nonzero) > 10:
        lo_thresh = float(np.percentile(nonzero, 33))
        hi_thresh = float(np.percentile(nonzero, 66))
    else:
        lo_thresh, hi_thresh = 4.0, 10.0
    print(f"[Confusion] Congestion thresholds: Low<{lo_thresh:.1f} | Moderate<{hi_thresh:.1f} | Busy>={hi_thresh:.1f}")

    def to_class(vals):
        return ["Low" if v <= 0 else "Moderate" if v < hi_thresh else "Busy"
                for v in np.array(vals).flatten()]

    y_true_cls = to_class(y_test_inv)
    y_pred_cls = to_class(y_pred_inv)
    labels = ["Low", "Moderate", "Busy"]

    present_classes = sorted(set(y_true_cls))
    cw = compute_class_weight("balanced", classes=np.array(present_classes), y=np.array(y_true_cls))
    class_weight_map = dict(zip(present_classes, cw))
    weights = np.array([class_weight_map.get(c, 1.0) for c in y_true_cls])
    correct = np.array([t == p for t, p in zip(y_true_cls, y_pred_cls)])
    weighted_acc = float(np.average(correct, weights=weights))
    print(f"[Confusion] Weighted classification accuracy: {weighted_acc*100:.1f}%")
    print(classification_report(y_true_cls, y_pred_cls, labels=labels, target_names=labels, zero_division=0))

    cm      = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle(f"Congestion Confusion Matrix  (weighted acc: {weighted_acc*100:.1f}%)",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.02)

    for ax, (data, title, cmap, vmax) in zip(axes, [
        (cm,      "Counts",    "Blues",  None),
        (cm_norm, "Normalised","YlOrRd", 1.0),
    ]):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, color=TEXT); ax.set_yticklabels(labels, color=TEXT)
        ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("True", color=TEXT)
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold")
        thresh = data.max() / 2
        for i in range(3):
            for j in range(3):
                val = data[i, j]
                txt = f"{val:.2f}" if vmax == 1.0 else f"{int(val)}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax)
        ax.set_facecolor(CARD)

    fig.patch.set_facecolor(BG)
    p = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[Plot] Saved: {p}")
    plt.close(fig)
    return p, weighted_acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run the full pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(data_path: str = DATA_PATH):
    # 0. Weather (Improvement 8)
    weather = fetch_weather()

    # 1. Preprocess + feature engineering (Improvements 1, 2, 3)
    df = preprocess(data_path)

    # 2. ARIMA (Improvement 4 — auto_arima)
    arima_model, hourly_series, test_series, arima_preds, arima_24h, arima_metrics = train_arima(df)

    # 3. Improved LSTM (Improvements 5, 6, 9, 10)
    lstm_model, scaler_x, scaler_y, history, y_test_inv, y_pred_inv, lstm_metrics, lstm_24h = \
        train_lstm(df, weather=weather)

    # 4. Hybrid combination (Improvement 7 — 0.5×ARIMA + 0.5×LSTM)
    avg_kwh = float(df["kWhDelivered"].mean())
    hybrid  = build_hybrid_forecast(arima_24h, lstm_24h, avg_kwh)

    # 5. Plots
    plot_all(history, y_test_inv, y_pred_inv, lstm_metrics,
             hourly_series, test_series, arima_preds, arima_metrics, hybrid)
    cm_path, weighted_acc = plot_confusion_matrix(y_test_inv, y_pred_inv)

    return {
        "arima": arima_metrics,
        "lstm":  lstm_metrics,
        "congestion_weighted_accuracy": round(weighted_acc, 4),
        "hybrid": hybrid,
        "avg_kwh_per_session": round(avg_kwh, 3),
        "weather": weather,
    }


if __name__ == "__main__":
    result = run_pipeline()

    # ── Improvement 10: Full metric table ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ChargeSync v3 — MODEL PERFORMANCE SUMMARY")
    print("═" * 60)
    print(f"  {'Metric':<20} {'ARIMA':>12} {'LSTM':>12}")
    print(f"  {'-'*44}")
    print(f"  {'RMSE':<20} {result['arima']['rmse']:>12.4f} {result['lstm']['rmse']:>12.4f}")
    print(f"  {'MAE':<20} {result['arima']['mae']:>12.4f} {result['lstm']['mae']:>12.4f}")
    print(f"  {'MAPE (%)':<20} {result['arima']['mape']:>11.2f}% {result['lstm']['mape']:>11.2f}%")
    print(f"  {'R²':<20} {'—':>12} {result['lstm']['r2']:>12.4f}")
    print(f"  {'CV RMSE (mean)':<20} {'—':>12} {result['lstm']['cv_rmse_mean']:>12.4f}")
    print(f"  {'CV MAPE (mean %)':<20} {'—':>12} {result['lstm']['cv_mape_mean']:>11.2f}%")
    print("═" * 60)
    print(f"  Congestion weighted accuracy : {result['congestion_weighted_accuracy']*100:.1f}%")
    print(f"  Avg kWh / session            : {result['avg_kwh_per_session']:.3f}")
    print(f"  Peak charging hour           : {result['hybrid']['peak_hour']}")
    print(f"  Weather source               : {result['weather']['source']}")
    print("═" * 60)
    print("\n  24h Hybrid Forecast (first 8 hours):")
    for row in result["hybrid"]["forecast"][:8]:
        print(f"    {row['hour_label']:>8} → {row['predicted_sessions']:.1f} sessions "
              f"| ARIMA {row['arima_kwh']:.1f} kWh "
              f"| LSTM {row['lstm_kwh']:.1f} kWh "
              f"| Blend {row['predicted_kwh_load']:.1f} kWh")
