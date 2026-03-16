"""
models.py — SQLAlchemy ORM table definitions for ChargeSync
Tables:
  ChargingSession  ← from acn_dataset.csv
  EVStation        ← from Indian_EV_Stations_Simplified.csv
  ForecastCache    ← stores latest ARIMA+LSTM result (replaces _hybrid_cache dict)
"""
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text, JSON
from database import Base


class ChargingSession(Base):
    """One row per EV charging session from the ACN dataset."""
    __tablename__ = "charging_sessions"

    id                = Column(Integer, primary_key=True, index=True)
    session_id        = Column(String, index=True, nullable=True)
    site_id           = Column(String, index=True, nullable=True)
    user_id           = Column(String, index=True, nullable=True)
    connection_time   = Column(DateTime(timezone=True), index=True, nullable=True)
    disconnect_time   = Column(DateTime(timezone=True), nullable=True)
    done_charging_time= Column(DateTime(timezone=True), nullable=True)
    kwh_delivered     = Column(Float, nullable=True)
    station_id        = Column(String, nullable=True)


class EVStation(Base):
    """One row per Indian EV charging station."""
    __tablename__ = "ev_stations"

    id           = Column(Integer, primary_key=True, index=True)
    station_name = Column(String, index=True)
    city         = Column(String, index=True, nullable=True)
    state        = Column(String, nullable=True)
    latitude     = Column(Float, nullable=False)
    longitude    = Column(Float, nullable=False)
    operator     = Column(String, nullable=True)
    connector_types = Column(String, nullable=True)
    is_active    = Column(Boolean, default=True)


class ForecastCache(Base):
    """Stores the latest ARIMA+LSTM hybrid forecast result."""
    __tablename__ = "forecast_cache"

    id         = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    result     = Column(JSON, nullable=False)   # full forecast dict
