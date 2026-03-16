"""
congestion_model.py  (v2 — XGBoost + KMeans clustering)
=========================================================
Predicts station occupancy congestion level using:
  - XGBoostClassifier  (Low / Moderate / Busy)  ← replaces profile lookup
  - KMeans clustering  (groups stations by usage pattern)
  - Statistical hourly profile baseline  (fast O(1) fallback)

Usage:
    from congestion_model import CongestionModel
    model = CongestionModel(data_path)
    model.train()
    result = model.predict("2-39-139-28", hours_ahead=1)
    clusters = model.get_station_clusters()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


# ─── Light in-memory cache ────────────────────────────────────────────────────
_prediction_cache: dict = {}


class CongestionModel:
    """
    Trains a station-level congestion model on historical ACN session data.

    Training pipeline:
      1. Load & preprocess sessions
      2. Build hourly occupancy profiles per station (fast fallback)
      3. Train XGBoost congestion classifier (Low/Moderate/Busy)
      4. Run KMeans to cluster stations by usage pattern
    """

    def __init__(self, data_path: str):
        self.data_path        = data_path
        self.df               = None
        self.station_profiles: dict = {}   # station_id → {(hour, dow) → avg_sessions}
        self.xgb_model        = None
        self.label_map        = {0: "Low", 1: "Moderate", 2: "Busy"}
        self.inv_label_map    = {"Low": 0, "Moderate": 1, "Busy": 2}
        self.kmeans           = None
        self.station_clusters: dict = {}   # station_id → cluster_id
        self.cluster_profiles: dict = {}   # cluster_id → avg hourly vector
        self.ready            = False

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Load & preprocess
    # ══════════════════════════════════════════════════════════════════════════
    def _load_data(self):
        df = pd.read_csv(self.data_path)
        df["connectionTime"]  = pd.to_datetime(df["connectionTime"],  utc=True, errors="coerce")
        df["disconnectTime"]  = pd.to_datetime(df["disconnectTime"],  utc=True, errors="coerce")
        df["doneChargingTime"]= pd.to_datetime(df.get("doneChargingTime"), utc=True, errors="coerce")
        df = df.dropna(subset=["connectionTime", "disconnectTime", "kWhDelivered"])
        df = df[df["kWhDelivered"] > 0].copy()

        # Feature columns
        df["hour"]         = df["connectionTime"].dt.hour
        df["dow"]          = df["connectionTime"].dt.dayofweek
        df["month"]        = df["connectionTime"].dt.month
        df["is_weekend"]   = (df["dow"] >= 5).astype(int)
        df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]      = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"]      = np.cos(2 * np.pi * df["dow"] / 7)
        df["month_sin"]    = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"]    = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        df["charging_duration"] = (
            (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600
        ).clip(lower=0)

        done_col = df["doneChargingTime"].fillna(df["disconnectTime"])
        df["idle_time"] = (
            (df["disconnectTime"] - done_col).dt.total_seconds() / 3600
        ).clip(lower=0)

        df["hour_ts"] = df["connectionTime"].dt.floor("h")
        self.df = df.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Hourly occupancy profiles (fast fallback baseline)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_profiles(self):
        if self.df is None:
            return

        records = []
        for _, row in self.df.iterrows():
            start = row["connectionTime"].floor("h")
            end   = row["disconnectTime"].ceil("h")
            slot  = start
            while slot < end:
                records.append({
                    "stationID":  row["stationID"],
                    "hour":       slot.hour,
                    "day_of_week":slot.dayofweek,
                })
                slot += timedelta(hours=1)

        slot_df     = pd.DataFrame(records)
        total_weeks = max(1, (self.df["connectionTime"].max() -
                              self.df["connectionTime"].min()).days / 7)

        profile = (
            slot_df.groupby(["stationID", "hour", "day_of_week"])
            .size()
            .reset_index(name="total_slots")
        )
        profile["avg_sessions"] = (profile["total_slots"] / total_weeks).round(2)

        for sid, grp in profile.groupby("stationID"):
            self.station_profiles[sid] = {
                (int(r["hour"]), int(r["day_of_week"])): float(r["avg_sessions"])
                for _, r in grp.iterrows()
            }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. XGBoost congestion classifier
    # ══════════════════════════════════════════════════════════════════════════
    def _train_xgboost(self):
        """
        Build an hourly occupancy dataset and train XGBoost to predict
        congestion level (Low / Moderate / Busy) per station-hour.
        """
        try:
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report

            if self.df is None or len(self.df) < 100:
                return

            # Build hourly occupancy counts per station
            hourly = (
                self.df.groupby(["stationID", "hour_ts"])
                .agg(
                    n_sessions     = ("kWhDelivered", "count"),
                    avg_kwh        = ("kWhDelivered", "mean"),
                    avg_duration   = ("charging_duration", "mean"),
                    avg_idle       = ("idle_time", "mean"),
                )
                .reset_index()
            )
            hourly["hour"]       = hourly["hour_ts"].dt.hour
            hourly["dow"]        = hourly["hour_ts"].dt.dayofweek
            hourly["month"]      = hourly["hour_ts"].dt.month
            hourly["is_weekend"] = (hourly["dow"] >= 5).astype(int)
            hourly["hour_sin"]   = np.sin(2 * np.pi * hourly["hour"] / 24)
            hourly["hour_cos"]   = np.cos(2 * np.pi * hourly["hour"] / 24)
            hourly["dow_sin"]    = np.sin(2 * np.pi * hourly["dow"] / 7)
            hourly["dow_cos"]    = np.cos(2 * np.pi * hourly["dow"] / 7)
            hourly["month_sin"]  = np.sin(2 * np.pi * (hourly["month"] - 1) / 12)
            hourly["month_cos"]  = np.cos(2 * np.pi * (hourly["month"] - 1) / 12)

            # Lag features per station
            hourly = hourly.sort_values(["stationID", "hour_ts"])
            hourly["lag_1"]  = hourly.groupby("stationID")["n_sessions"].shift(1).fillna(0)
            hourly["lag_24"] = hourly.groupby("stationID")["n_sessions"].shift(24).fillna(0)
            hourly["roll_6"] = (
                hourly.groupby("stationID")["n_sessions"]
                .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
                .fillna(0)
            )

            # Label: congestion class from session count percentiles
            q33 = hourly["n_sessions"].quantile(0.33)
            q66 = hourly["n_sessions"].quantile(0.66)
            hourly["label"] = hourly["n_sessions"].apply(
                lambda v: 0 if v <= q33 else (1 if v <= q66 else 2)
            )

            features = [
                "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos", "is_weekend",
                "avg_kwh", "avg_duration", "avg_idle",
                "lag_1", "lag_24", "roll_6",
            ]
            X = hourly[features].fillna(0).values
            y = hourly["label"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Encode station ID for cluster feature (added after split)
            self._xgb_features = features

            clf = xgb.XGBClassifier(
                n_estimators       = 300,
                max_depth          = 6,
                learning_rate      = 0.05,
                subsample          = 0.8,
                colsample_bytree   = 0.8,
                use_label_encoder  = False,
                eval_metric        = "mlogloss",
                tree_method        = "hist",
                random_state       = 42,
                verbosity          = 0,
            )
            clf.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = clf.predict(X_test)
            acc    = float((y_pred == y_test).mean())
            print(f"[CongestionModel] XGBoost accuracy: {acc*100:.1f}%")
            print(classification_report(
                y_test, y_pred,
                target_names=["Low", "Moderate", "Busy"],
                zero_division=0,
            ))

            self.xgb_model    = clf
            self._congestion_q33 = q33
            self._congestion_q66 = q66

        except Exception as e:
            print(f"[CongestionModel] XGBoost training skipped: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. KMeans station clustering by usage pattern
    # ══════════════════════════════════════════════════════════════════════════
    def _train_kmeans(self, n_clusters: int = 4):
        """
        Groups stations into n_clusters based on their hourly demand pattern
        (24-vector of average sessions per hour-of-day, averaged over all dow).
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            if self.df is None or len(self.df) < 50:
                return

            # Build 24-dim hourly usage vector per station
            usage = (
                self.df.groupby(["stationID", "hour"])
                .size()
                .reset_index(name="count")
            )
            pivot = usage.pivot(index="stationID", columns="hour", values="count").fillna(0)
            # Fill any missing hour columns
            for h in range(24):
                if h not in pivot.columns:
                    pivot[h] = 0
            pivot = pivot[list(range(24))]

            stations = pivot.index.tolist()
            X        = pivot.values.astype(float)

            scaler  = StandardScaler()
            X_sc    = scaler.fit_transform(X)

            k       = min(n_clusters, len(stations))
            km      = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels  = km.fit_predict(X_sc)

            for sid, cluster_id in zip(stations, labels):
                self.station_clusters[sid] = int(cluster_id)

            # Compute cluster-level average hourly profiles (unscaled, for display)
            for c in range(k):
                mask = labels == c
                self.cluster_profiles[c] = pivot.values[mask].mean(axis=0).tolist()

            self.kmeans = km
            print(f"[CongestionModel] KMeans: {k} clusters | "
                  f"Station counts: { {c: int((labels==c).sum()) for c in range(k)} }")

        except Exception as e:
            print(f"[CongestionModel] KMeans skipped: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Public: train
    # ══════════════════════════════════════════════════════════════════════════
    def train(self):
        try:
            print("[CongestionModel] Loading data …")
            self._load_data()
            print("[CongestionModel] Building profiles …")
            self._build_profiles()
            print("[CongestionModel] Training XGBoost classifier …")
            self._train_xgboost()
            print("[CongestionModel] Running KMeans clustering …")
            self._train_kmeans(n_clusters=4)
            self.ready = True
            print("[CongestionModel] Ready ✓")
        except Exception as e:
            print(f"[CongestionModel] Training error: {e}")
            self.ready = True   # serve baseline anyway

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Public: predict (per station)
    # ══════════════════════════════════════════════════════════════════════════
    def predict(self, station_id: str, hours_ahead: int = 1) -> dict:
        """
        Predicts congestion for `station_id` at now + hours_ahead.
        Uses XGBoost when available, falls back to statistical profile.
        """
        target_dt = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)
        hour = target_dt.hour
        dow  = target_dt.weekday()

        # ── XGBoost prediction ────────────────────────────────────────────────
        xgb_level = None
        if self.xgb_model is not None and self.df is not None:
            try:
                st_df = self.df[self.df["stationID"] == station_id]
                lag1  = float(st_df.groupby(st_df["hour_ts"].dt.floor("h")).size().iloc[-1]) \
                        if len(st_df) > 0 else 0.0
                lag24 = float(
                    st_df[st_df["hour"] == hour]["kWhDelivered"].count()
                ) if len(st_df) > 0 else 0.0
                roll6 = lag1  # simplified: last known count

                feat = np.array([[
                    np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                    np.sin(2*np.pi*dow/7),   np.cos(2*np.pi*dow/7),
                    np.sin(2*np.pi*(target_dt.month-1)/12),
                    np.cos(2*np.pi*(target_dt.month-1)/12),
                    int(dow >= 5),
                    float(st_df["kWhDelivered"].mean() if len(st_df) > 0 else 5.0),
                    float(st_df["charging_duration"].mean() if len(st_df) > 0 else 1.0),
                    float(st_df["idle_time"].mean() if len(st_df) > 0 else 0.0),
                    lag1, lag24, roll6,
                ]])
                pred_class = int(self.xgb_model.predict(feat)[0])
                xgb_level  = self.label_map[pred_class]
            except Exception:
                pass

        # ── Profile baseline ──────────────────────────────────────────────────
        profile     = self.station_profiles.get(station_id, {})
        avg_sessions = profile.get((hour, dow), None)
        if avg_sessions is None:
            fallback = [v for (h, d), v in profile.items() if d == dow]
            avg_sessions = float(np.mean(fallback)) if fallback else 1.0

        # ── Total charger estimate ────────────────────────────────────────────
        if self.df is not None:
            st_df = self.df[self.df["stationID"] == station_id]
            total_chargers = max(2, int(
                st_df.shape[0] / max(1, len(self.df["connectionTime"].dt.date.unique()))
            ))
        else:
            total_chargers = 4

        occupied       = min(round(avg_sessions), total_chargers)
        available      = max(0, total_chargers - occupied)
        congestion_pct = int((occupied / max(1, total_chargers)) * 100)

        # Use XGBoost level if available, else derive from pct
        if xgb_level:
            level = xgb_level
        elif congestion_pct < 40:
            level = "Low"
        elif congestion_pct < 70:
            level = "Moderate"
        else:
            level = "Busy"

        predicted_wait = 0 if available > 0 else int(occupied * 12)

        # 6-hour sparkline
        sparkline = []
        for h_offset in range(6):
            future_dt = target_dt + timedelta(hours=h_offset)
            fh, fd    = future_dt.hour, future_dt.weekday()
            fv  = profile.get((fh, fd), avg_sessions)
            fpct = int(min(100, (fv / max(1, total_chargers)) * 100))
            sparkline.append(fpct)

        return {
            "station_id":               station_id,
            "predicted_at_hour":        hour,
            "total_chargers":           total_chargers,
            "predicted_active_sessions":occupied,
            "available_chargers":       available,
            "congestion_pct":           congestion_pct,
            "predicted_wait_minutes":   predicted_wait,
            "level":                    level,
            "xgb_level":                xgb_level,           # raw XGBoost output
            "cluster_id":               self.station_clusters.get(station_id, -1),
            "sparkline_6h":             sparkline,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 7. Public: predict_all
    # ══════════════════════════════════════════════════════════════════════════
    def predict_all(self, hours_ahead: int = 1) -> list:
        """Predict for every known station."""
        if self.df is None:
            return []
        return [self.predict(sid, hours_ahead)
                for sid in self.df["stationID"].unique().tolist()]

    # ══════════════════════════════════════════════════════════════════════════
    # 8. Public: get_station_clusters
    # ══════════════════════════════════════════════════════════════════════════
    def get_station_clusters(self) -> dict:
        """
        Returns clustering result:
          {
            "station_clusters": {station_id: cluster_id, ...},
            "cluster_profiles": {cluster_id: [24 avg values], ...},
            "n_clusters": int,
          }
        """
        return {
            "station_clusters": self.station_clusters,
            "cluster_profiles": self.cluster_profiles,
            "n_clusters":       len(self.cluster_profiles),
        }
