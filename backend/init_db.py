"""
init_db.py — One-time database seeding script.
Run this ONCE to create all tables and import both CSV files.

Usage:
    python init_db.py

Safe to re-run: skips import if rows already exist.
"""
import os
import sys
import pandas as pd
from datetime import timezone
from sqlalchemy.orm import Session

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))

from database import engine, SessionLocal
from models import Base, ChargingSession, EVStation

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
ACN_CSV         = os.path.join(BASE_DIR, "acn_dataset.csv")
STATIONS_CSV    = os.path.join(BASE_DIR, "Indian_EV_Stations_Simplified.csv")
BATCH_SIZE      = 1000   # rows per DB commit


def create_tables():
    print("[DB] Creating tables ...")
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables ready.")


def _naive(ts):
    """Convert a pandas Timestamp (possibly tz-aware) to a naive Python datetime or None."""
    if ts is None or (hasattr(ts, 'isnull') and ts.isnull()):
        return None
    try:
        import pandas as pd
        if pd.isnull(ts):
            return None
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            return ts.to_pydatetime().replace(tzinfo=None)
        return ts.to_pydatetime()
    except Exception:
        return None


def seed_charging_sessions(db: Session):
    existing = db.query(ChargingSession).count()
    if existing > 0:
        print(f"[DB] ChargingSession: {existing:,} rows already present — skipping import.")
        return

    print(f"[DB] Importing ChargingSession from {ACN_CSV} ...")
    df = pd.read_csv(ACN_CSV)

    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df = df.dropna(subset=["connectionTime", "kWhDelivered"])
    df = df[df["kWhDelivered"] > 0].reset_index(drop=True)

    total = len(df)
    print(f"[DB]   {total:,} valid sessions to insert ...")

    batch = []
    for i, row in df.iterrows():
        batch.append(ChargingSession(
            session_id         = str(row.get("sessionID", "")),
            site_id            = str(row.get("siteID", "")),
            user_id            = str(row.get("userID", "")),
            connection_time    = _naive(row.get("connectionTime")),
            disconnect_time    = _naive(row.get("disconnectTime")),
            done_charging_time = _naive(row.get("doneChargingTime")),
            kwh_delivered      = float(row["kWhDelivered"]),
            station_id         = str(row.get("stationID", "")),
        ))
        if len(batch) >= BATCH_SIZE:
            db.bulk_save_objects(batch)
            db.commit()
            batch = []
            print(f"[DB]   ... {i+1:,}/{total:,} sessions committed", end="\r")

    if batch:
        db.bulk_save_objects(batch)
        db.commit()

    count = db.query(ChargingSession).count()
    print(f"\n[DB] ChargingSession: {count:,} rows inserted. Done.")



def seed_ev_stations(db: Session):
    existing = db.query(EVStation).count()
    if existing > 0:
        print(f"[DB] EVStation: {existing:,} rows already present — skipping import.")
        return

    print(f"[DB] Importing EVStation from {STATIONS_CSV} ...")
    df = pd.read_csv(STATIONS_CSV).fillna("")

    batch = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            lat = float(row.get("Latitude", 0))
            lng = float(row.get("Longitude", 0))
            if lat == 0 or lng == 0:
                skipped += 1
                continue
            batch.append(EVStation(
                station_name    = str(row.get("Station Name", "")),
                city            = str(row.get("City", "")),
                state           = str(row.get("State", "")),
                latitude        = lat,
                longitude       = lng,
                operator        = str(row.get("Operator", "")),
                connector_types = str(row.get("Connector Types", "")),
                is_active       = True,
            ))
        except Exception:
            skipped += 1
            continue

    db.bulk_save_objects(batch)
    db.commit()

    count = db.query(EVStation).count()
    print(f"[DB] EVStation: {count:,} rows inserted ({skipped} skipped). Done.")


if __name__ == "__main__":
    create_tables()
    db = SessionLocal()
    try:
        seed_charging_sessions(db)
        seed_ev_stations(db)
        print("\n[DB] Database seeding complete!")
        print(f"[DB] Database file: {os.path.abspath('chargesync.db')}")
    finally:
        db.close()
