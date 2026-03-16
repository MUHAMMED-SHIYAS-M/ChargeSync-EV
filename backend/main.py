from fastapi import FastAPI, Query, Body, Depends
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from data_processor import load_and_preprocess, get_global_stats
from scheduler import ChargingScheduler
from datetime import datetime, timezone
import os
import requests
import math
import hashlib

# ── Database ──────────────────────────────────────────────────────────────────
from database import SessionLocal, engine
from models import Base, ChargingSession, EVStation, ForecastCache
import pandas as pd

# Ensure all tables exist on startup
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY" # REPLACE WITH YOUR ACTUAL API KEY

app = FastAPI(title="EV ChargeSync API")

class TripPlanRequest(BaseModel):
    source: str
    destination: str

def geocode_nominatim(address: str):
    # Check if address is already lat,lng
    try:
        parts = address.split(',')
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except:
        pass
        
    url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
    headers = {'User-Agent': 'ChargeSyncApp/1.0'}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data and len(data) > 0:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        print(f"Geocoding error for {address}: {e}")
        pass
    
    # Fallback to some Indian cities if failed or rate limited
    lower_addr = address.lower()
    if "mumbai" in lower_addr: return 19.0760, 72.8777
    if "delhi" in lower_addr: return 28.7041, 77.1025
    if "bangalore" in lower_addr or "bengaluru" in lower_addr: return 12.9716, 77.5946
    if "pune" in lower_addr: return 18.5204, 73.8567
    if "chennai" in lower_addr: return 13.0827, 80.2707
    if "hyderabad" in lower_addr: return 17.3850, 78.4867
    
    return None, None

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from lstm_service import train_and_predict_demand
from congestion_model import CongestionModel
from session_simulator import SessionSimulator
from contextlib import asynccontextmanager

DATA_PATH = r"d:\games\group11ziipp\group11\backend\acn-data (1) (1).csv"

# ── Load training DataFrame from DB (fallback to CSV if DB empty) ─────────────
_db_startup = SessionLocal()
try:
    _row_count = _db_startup.query(ChargingSession).count()
finally:
    _db_startup.close()

if _row_count > 0:
    print(f"[Startup] Loading {_row_count:,} sessions from database ...")
    _db_load = SessionLocal()
    try:
        rows = _db_load.query(
            ChargingSession.connection_time,
            ChargingSession.disconnect_time,
            ChargingSession.done_charging_time,
            ChargingSession.kwh_delivered,
            ChargingSession.station_id,
            ChargingSession.user_id,
            ChargingSession.site_id,
        ).all()
        df = pd.DataFrame(rows, columns=[
            "connectionTime", "disconnectTime", "doneChargingTime",
            "kWhDelivered", "stationID", "userID", "siteID"
        ])
        for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        print(f"[Startup] DataFrame ready: {len(df):,} rows.")
    finally:
        _db_load.close()
else:
    print("[Startup] DB empty — falling back to CSV.")
    df = load_and_preprocess(DATA_PATH)
scheduler = ChargingScheduler(df)

# Train congestion model on startup
congestion_model = CongestionModel(DATA_PATH)
congestion_model.train()

# Create simulator (started in lifespan below)
session_sim = SessionSimulator(congestion_model)

@asynccontextmanager
async def lifespan(app_):
    # Start real-time session simulator in background
    session_sim.start()
    yield
    session_sim.stop()

app.router.lifespan_context = lifespan

@app.get("/stations")
def get_stations():
    stations = df['stationID'].unique().tolist()
    return {"stations": stations}

@app.get("/stats")
def get_stats():
    stats = get_global_stats(df)
    return stats

@app.get("/predict")
def predict(station_id: str, arrival_time: str = None):
    if arrival_time is None:
        arrival_dt = datetime.now(timezone.utc)
    else:
        try:
            arrival_dt = datetime.fromisoformat(arrival_time.replace('Z', '+00:00'))
            if arrival_dt.tzinfo is None:
                arrival_dt = arrival_dt.replace(tzinfo=timezone.utc)
        except:
            arrival_dt = datetime.now(timezone.utc) # Fallback
            
    wait_time = scheduler.predict_wait_time(station_id, arrival_dt)
    return {
        "stationID": station_id,
        "arrivalTime": arrival_dt.isoformat(),
        "predictedWaitMinutes": round(wait_time, 2),
        "status": "Busy" if wait_time > 5 else "Available"
    }

@app.get("/optimize")
def optimize(battery: float = 20.0, lat: float = None, lng: float = None):
    """
    Returns top EV charging station recommendations from the real Indian EV dataset,
    enriched with live congestion data. Optionally filtered by proximity to lat/lng.
    """
    import pandas as pd, math, hashlib

    try:
        CSV_PATH = r"d:\games\group11ziipp\group11\backend\Indian_EV_Stations_Simplified.csv"
        df_ev = pd.read_csv(CSV_PATH).fillna("")

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            dLat = math.radians(lat2 - lat1)
            dLon = math.radians(lon2 - lon1)
            a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        results = []
        for idx, row in df_ev.iterrows():
            try:
                st_lat = float(row["Latitude"])
                st_lng = float(row["Longitude"])
            except Exception:
                continue

            dist_km = None
            if lat is not None and lng is not None:
                dist_km = round(haversine(lat, lng, st_lat, st_lng), 2)
                if dist_km > 50:
                    continue  # skip stations more than 50 km away

            station_name = str(row.get("Station Name", "")).strip() or f"Station-{idx}"
            city = str(row.get("City", "")).strip()
            state = str(row.get("State", "")).strip()
            operator = str(row.get("Operator", "")).strip()
            charger_type = str(row.get("Charger Type", "AC")).strip()
            connector = str(row.get("Connector Type", "")).strip()

            # Try to parse power
            try:
                power_kw = float(str(row.get("Power (kW)", "22")).replace("kW", "").strip())
            except Exception:
                power_kw = 22.0

            # Deterministic availability from name hash
            hash_val = int(hashlib.md5(station_name.encode()).hexdigest(), 16)
            total_chargers = (hash_val % 7) + 2
            active_sessions = hash_val % (total_chargers + 1)

            # Pull live congestion from session simulator if possible
            try:
                all_acn = session_sim.get_all_states()
                if all_acn:
                    sample = all_acn[hash(station_name) % len(all_acn)]
                    occupancy_pct = sample["occupancy_pct"]
                    total_chargers = max(total_chargers, sample["total_chargers"])
                    active_sessions = round((occupancy_pct / 100) * total_chargers)
                else:
                    occupancy_pct = round((active_sessions / total_chargers) * 100)
            except Exception:
                occupancy_pct = round((active_sessions / total_chargers) * 100)

            available = max(0, total_chargers - active_sessions)
            wait_mins = 0 if available > 0 else int(active_sessions * 10)
            congestion_level = "Low" if occupancy_pct < 40 else "Moderate" if occupancy_pct < 70 else "Busy"

            # Estimate travel time (assume 40 km/h in city if location given)
            travel_time = round(dist_km * 1.5, 1) if dist_km is not None else 0.0

            # Score: higher power, lower congestion, higher availability = better
            score = round(
                100
                - (occupancy_pct * 0.4)
                + (available * 5)
                + min(power_kw / 10, 10)   # power bonus, capped at 10
                - (travel_time * 0.5 if travel_time else 0),
                1
            )

            results.append({
                "stationID": station_name,
                "city": city,
                "state": state,
                "operator": operator,
                "chargerType": charger_type,
                "connectorType": connector,
                "powerKw": power_kw,
                "latitude": st_lat,
                "longitude": st_lng,
                "distanceKm": dist_km,
                "travelTime": travel_time,
                "predictedWait": wait_mins,
                "availableChargers": available,
                "totalChargers": total_chargers,
                "occupancyPct": occupancy_pct,
                "congestionLevel": congestion_level,
                "score": score,
            })

        # Sort by score descending, return top 8
        results.sort(key=lambda x: -x["score"])
        return results[:8]

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

@app.get("/predict-demand")
def predict_demand():
    """
    Triggers the LSTM model training and prediction on the dataset.
    Returns RMSE and the latest prediction comparison.
    """
    try:
        result = train_and_predict_demand(DATA_PATH)
        return result
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}


@app.get("/stations/congestion")
def get_station_congestion(station_id: str = None, hours_ahead: int = 0):
    """
    Returns LIVE simulated congestion state for a station (or all stations).
    Falls back to LSTM profile prediction if simulator hasn't initialised yet.
    """
    try:
        if station_id:
            state = session_sim.get_enriched_state(station_id)
            return state
        else:
            all_states = session_sim.get_all_states()
            # Enrich each with sparkline
            enriched = [session_sim.get_enriched_state(s["station_id"]) for s in all_states]
            return {"congestion": enriched}
    except Exception as e:
        return {"error": str(e)}


@app.get("/stations/congestion/forecast")
def get_congestion_forecast():
    """
    Returns a 6-hour congestion forecast across all ACN stations,
    aggregated by hour. Used by the Dashboard Forecast panel.
    """
    try:
        from datetime import datetime, timezone, timedelta

        all_states = session_sim.get_all_states()
        hourly_totals = [0] * 6
        hourly_capacity = [0] * 6

        for s in all_states:
            sid = s["station_id"]
            capacity = s["total_chargers"]
            pred = congestion_model.predict(sid, hours_ahead=0)
            sparkline = pred.get("sparkline_6h", [s["congestion_pct"]] * 6)
            for i, pct in enumerate(sparkline):
                hourly_totals[i] += pct * capacity
                hourly_capacity[i] += capacity * 100

        hourly_avg_pct = [
            int(hourly_totals[i] / max(1, hourly_capacity[i]) * 100)
            for i in range(6)
        ]

        now = datetime.now(timezone.utc)
        labels = [(now + timedelta(hours=i)).strftime("%I %p") for i in range(6)]

        return {"labels": labels, "congestion_pct": hourly_avg_pct}
    except Exception as e:
        return {"error": str(e)}


@app.get("/stations/smart-routes")
def get_smart_routes(lat: float, lng: float, radius: float = 10.0, limit: int = 8):
    """
    Returns nearby EV stations ranked by a composite score:
        score = 100 - (drive_mins * 1.5) - (congestion_pct * 0.4) + (available * 3)
    Each result includes a real OSRM driving route polyline for map drawing.
    """
    import math, requests as req
    import pandas as pd

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    try:
        CSV_PATH = r"d:\games\group11ziipp\group11\backend\Indian_EV_Stations_Simplified.csv"
        df_ev = pd.read_csv(CSV_PATH)
        df_ev = df_ev.dropna(subset=["Latitude", "Longitude"])

        # Filter by radius
        df_ev["dist_km"] = df_ev.apply(
            lambda r: haversine(lat, lng, float(r["Latitude"]), float(r["Longitude"])), axis=1
        )
        nearby = df_ev[df_ev["dist_km"] <= radius].sort_values("dist_km").head(limit * 3)

        results = []
        for _, row in nearby.iterrows():
            st_lat = float(row["Latitude"])
            st_lng = float(row["Longitude"])
            dist_km = round(float(row["dist_km"]), 2)

            # Get OSRM driving route
            drive_mins = dist_km * 2.5  # fallback estimate
            polyline = [[lat, lng], [st_lat, st_lng]]
            try:
                osrm_url = (
                    f"http://router.project-osrm.org/route/v1/driving/"
                    f"{lng},{lat};{st_lng},{st_lat}"
                    f"?overview=full&geometries=geojson&steps=false"
                )
                r = req.get(osrm_url, timeout=4)
                if r.status_code == 200:
                    data = r.json()
                    route_data = data["routes"][0]
                    drive_mins = round(route_data["duration"] / 60, 1)
                    coords = route_data["geometry"]["coordinates"]
                    polyline = [[c[1], c[0]] for c in coords]  # OSRM gives [lng,lat]
            except Exception:
                pass

            # Live congestion from simulator
            # Indian EV stations don't have ACN IDs; derive congestion from station data
            # Use available chargers field or simulate based on time-of-day profile
            charger_type = str(row.get("Charger Type", "AC"))
            total_chargers = 4 if "DC" in charger_type.upper() else 2

            # Try to match an ACN station by time-of-day occupancy pattern
            avg_occupancy_pct = 35  # default low
            try:
                from datetime import datetime, timezone
                hour = datetime.now(timezone.utc).hour
                dow = datetime.now(timezone.utc).weekday()
                # Sample occupancy from nearby ACN states as a proxy for Indian stations
                all_acn = session_sim.get_all_states()
                if all_acn:
                    sample = all_acn[hash(str(row.get("Station Name", ""))) % len(all_acn)]
                    avg_occupancy_pct = sample["occupancy_pct"]
                    total_chargers = max(total_chargers, sample["total_chargers"])
            except Exception:
                pass

            active = round((avg_occupancy_pct / 100) * total_chargers)
            available = max(0, total_chargers - active)
            level = "Low" if avg_occupancy_pct < 40 else "Moderate" if avg_occupancy_pct < 70 else "Busy"
            wait_mins = 0 if available > 0 else int(active * 10)

            # Composite score
            score = round(
                100
                - (drive_mins * 1.5)
                - (avg_occupancy_pct * 0.4)
                + (available * 3),
                1
            )

            results.append({
                "name": str(row.get("Station Name", "EV Station")),
                "city": str(row.get("City", "")),
                "state": str(row.get("State", "")),
                "latitude": st_lat,
                "longitude": st_lng,
                "distance_km": dist_km,
                "drive_minutes": drive_mins,
                "total_chargers": total_chargers,
                "available_chargers": available,
                "congestion_pct": avg_occupancy_pct,
                "congestion_level": level,
                "predicted_wait_minutes": wait_mins,
                "score": score,
                "route_polyline": polyline,
            })

        # Sort by score descending, take top `limit`
        results.sort(key=lambda x: -x["score"])
        results = results[:limit]

        return {"routes": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}



@app.get("/places/nearby")
def get_nearby_places(lat: float, lng: float, radius: int = 5000, type: str = "charging_station"):
    """
    Proxies the Google Places Nearby Search API.
    """
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        return {"error": "Google API Key not configured in backend/main.py"}
        
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&type={type}&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": f"Failed to fetch from Google Maps API: {str(e)}"}

    except Exception as e:
        return {"error": f"Failed to fetch from Google Maps API: {str(e)}"}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of the Earth in km
    import math
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.get("/stations/nearby")
def get_nearby_stations_csv(lat: float, lng: float, radius: float = 10.0):
    try:
        import pandas as pd
        import numpy as np
        df_map = pd.read_csv(r"d:\games\group11ziipp\group11\backend\Indian_EV_Stations_Simplified.csv")
        df_map = df_map.fillna("")
        
        nearby_stations = []
        for idx, row in df_map.iterrows():
            try:
                st_lat = float(row['Latitude'])
                st_lon = float(row['Longitude'])
                if np.isnan(st_lat) or np.isnan(st_lon):
                    continue
                
                dist = haversine_distance(lat, lng, st_lat, st_lon)
                if dist <= radius:
                    # Mock availability data deterministically based on station name
                    station_name = str(row['Station Name']) or f"Station-{idx}"
                    hash_val = int(hashlib.md5(station_name.encode()).hexdigest(), 16)
                    
                    total_chargers = (hash_val % 7) + 2 # 2 to 8 chargers
                    active_sessions = (hash_val % (total_chargers + 1))
                    available = total_chargers - active_sessions
                    
                    charger_type = str(row.get('Connector Type', 'AC/DC Fast'))
                    if charger_type == "": charger_type = "CCS (DC Fast)"
                    
                    nearby_stations.append({
                        "id": f"st-{idx}",
                        "name": station_name,
                        "distance": round(dist, 2),
                        "latitude": st_lat,
                        "longitude": st_lon,
                        "total_chargers": total_chargers,
                        "active_sessions": active_sessions,
                        "available": available,
                        "status": "Available" if available > 0 else "Occupied",
                        "charger_type": charger_type,
                        "wait_time_mins": (active_sessions * 15) if available == 0 else 0
                    })
            except Exception:
                continue
                
        # Sort by distance nearest to furthest
        nearby_stations.sort(key=lambda x: x['distance'])
        return {"stations": nearby_stations}
    except Exception as e:
        return {"error": str(e)}

class TripPlanRequest(BaseModel):
    source: str
    destination: str
    optimize_for_wait: bool = False  # If True, re-rank stops by min wait rather than min distance
    battery_pct: float = 80.0        # Current battery level as a percentage (0-100)
    vehicle_range_km: float = 400.0  # Full range of the vehicle on a 100% charge (km)

@app.post("/trip/plan")
def plan_trip(request: TripPlanRequest):
    import pandas as pd
    
    source_lat, source_lng = geocode_nominatim(request.source)
    dest_lat, dest_lng = geocode_nominatim(request.destination)
    
    if source_lat is None or dest_lat is None:
        return {"error": "Could not geocode source or destination. Ensure city names are properly spelled."}

    # Fetch real route using OSRM
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{source_lng},{source_lat};{dest_lng},{dest_lat}?geometries=geojson&overview=full"
    
    route_coords = []
    try:
        resp = requests.get(osrm_url)
        data = resp.json()
        if data.get('code') == 'Ok':
            # GeoJSON returns [lng, lat], Leaflet wants [lat, lng]
            coords = data['routes'][0]['geometry']['coordinates']
            route_coords = [[lat, lng] for lng, lat in coords]
    except Exception as e:
        print("OSRM error:", e)
        
    # Fallback to straight line if OSRM fails
    if not route_coords:
        route_coords = [[source_lat, source_lng], [dest_lat, dest_lng]]

    # Load Indian EV stations from DATABASE instead of CSV
    suggested = []
    try:
        _db = SessionLocal()
        try:
            # Bounding box filter in DB query
            min_lat = min(source_lat, dest_lat) - 0.5
            max_lat = max(source_lat, dest_lat) + 0.5
            min_lng = min(source_lng, dest_lng) - 0.5
            max_lng = max(source_lng, dest_lng) + 0.5

            station_rows = _db.query(EVStation).filter(
                EVStation.latitude  >= min_lat,
                EVStation.latitude  <= max_lat,
                EVStation.longitude >= min_lng,
                EVStation.longitude <= max_lng,
                EVStation.is_active == True,
            ).all()
        finally:
            _db.close()

        df_map_rows = [{
            "Station Name": s.station_name,
            "City":         s.city,
            "Latitude":     s.latitude,
            "Longitude":    s.longitude,
        } for s in station_rows]

        def point_to_segment_dist(px, py, x1, y1, x2, y2):
            # Distance from point (px,py) to line segment (x1,y1)-(x2,y2) 
            # Note: Approximating distance purely in degrees for speed
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return np.hypot(px - x1, py - y1)
            t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            return np.hypot(px - closest_x, py - closest_y)
            
        MAX_DIST_DEGREES = 0.05 # Roughly 5km depending on latitude
        
        for idx, row in enumerate(df_map_rows):
            try:
                lat = float(row['Latitude'])
                lng = float(row['Longitude'])
                if np.isnan(lat) or np.isnan(lng):
                    continue

                # Verify proximity to the actual polyline
                is_near_route = False
                for i in range(len(route_coords) - 1):
                    segment_start = route_coords[i]
                    segment_end = route_coords[i+1]
                    dist = point_to_segment_dist(lat, lng, segment_start[0], segment_start[1], segment_end[0], segment_end[1])
                    if dist <= MAX_DIST_DEGREES:
                        is_near_route = True
                        break

                if is_near_route:
                    dist_to_start = haversine_distance(source_lat, source_lng, lat, lng)
                    suggested.append({
                        "stationID":     str(row['Station Name']) or f"Station-{idx}",
                        "latitude":      lat,
                        "longitude":     lng,
                        "city":          str(row['City']),
                        "status":        "Available",
                        "dist_to_start": dist_to_start
                    })
            except Exception:
                continue
                
        # Sort suggested stations by distance to start point so the nearest is first
        suggested.sort(key=lambda x: x.get('dist_to_start', float('inf')))
        
        # To avoid overcrowding, let's limit to 20 stations
        suggested = suggested[:20]
        
        # Enrich each suggested station with congestion prediction
        for station in suggested:
            try:
                sid = station.get("stationID", "")
                cong = congestion_model.predict(sid, hours_ahead=1)
                station["congestion_pct"] = cong.get("congestion_pct", 0)
                station["predicted_wait_minutes"] = cong.get("predicted_wait_minutes", 0)
                station["congestion_level"] = cong.get("level", "Low")
                station["available_chargers"] = cong.get("available_chargers", 1)
                station["total_chargers"] = cong.get("total_chargers", 4)
                # Optimization score: blend distance and congestion
                dist_score = station.get("dist_to_start", 0)
                wait_score = cong.get("predicted_wait_minutes", 0)
                station["optimization_score"] = round(
                    100 - (0.5 * min(dist_score, 100)) - (0.5 * min(wait_score, 60)), 1
                )
            except Exception:
                station.setdefault("congestion_pct", 0)
                station.setdefault("predicted_wait_minutes", 0)
                station.setdefault("congestion_level", "Low")
                station.setdefault("optimization_score", 50.0)

        # Re-rank based on optimize_for_wait flag
        if request.optimize_for_wait:
            # Sort by lowest predicted wait time (least congested first)
            suggested.sort(key=lambda x: x.get("predicted_wait_minutes", float("inf")))
            sort_mode = "wait_time"
        else:
            # Sort by closest distance to trip start
            suggested.sort(key=lambda x: x.get("dist_to_start", float("inf")))
            sort_mode = "distance"

        # Tag each station with how it was sorted (useful for frontend labels)
        for s in suggested:
            s["sort_mode"] = sort_mode

        # ── Battery-aware reachability annotation ─────────────────────────
        battery_pct = max(0.0, min(100.0, request.battery_pct))
        vehicle_range_km = max(1.0, request.vehicle_range_km)
        reachable_km = round((battery_pct / 100.0) * vehicle_range_km, 1)

        for station in suggested:
            dist = station.get("dist_to_start", 0)
            station["km_to_station"] = round(dist, 1)
            station["reachable"] = dist <= reachable_km

    except Exception as e:
        print("Error matching stations to route:", e)
        reachable_km = round((request.battery_pct / 100.0) * request.vehicle_range_km, 1)

    return {
        "route": route_coords,
        "suggested_stations": suggested,
        "reachable_km": reachable_km
    }

@app.get("/vehicles")
def get_vehicles():
    try:
        import pandas as pd
        car_df = pd.read_csv(r"d:\games\group11ziipp\group11\backend\ElectricCarData_Clean (1).csv")
        car_df = car_df.fillna("")
        
        vehicles_dict = {}
        for _, row in car_df.iterrows():
            brand = str(row['Brand']).strip()
            model = str(row['Model']).strip()
            vehicles_dict[(brand, model)] = {
                "model": model,
                "accel": row['AccelSec'] if row['AccelSec'] != '' else None,
                "topSpeed": row['TopSpeed_KmH'] if row['TopSpeed_KmH'] != '' else None,
                "range": row['Range_Km'] if row['Range_Km'] != '' else None,
                "efficiency": row['Efficiency_WhKm'] if row['Efficiency_WhKm'] != '' else None,
                "fastCharge": row['FastCharge_KmH'] if row['FastCharge_KmH'] != '' else None,
                "price": row['PriceEuro'] if row['PriceEuro'] != '' else None
            }
            
        try:
            extra_df = pd.read_csv(r"d:\games\group11ziipp\group11\backend\EV_cars.csv")
            extra_df = extra_df.fillna("")
            for _, row in extra_df.iterrows():
                brand = str(row['Brand']).strip()
                model = str(row['Model']).strip()
                prev = vehicles_dict.get((brand, model), {})
                
                vehicles_dict[(brand, model)] = {
                    "model": model,
                    "accel": prev.get("accel", None),
                    "topSpeed": prev.get("topSpeed", None),
                    "range": row['Range_Km'] if row['Range_Km'] != '' else prev.get("range", None),
                    "efficiency": prev.get("efficiency", None),
                    "fastCharge": prev.get("fastCharge", None),
                    "price": prev.get("price", None)
                }
        except Exception as e:
            print("Error loading EV_cars.csv:", e)

        brands = {}
        for (brand, model), data in vehicles_dict.items():
            if brand not in brands:
                brands[brand] = []
            brands[brand].append(data)
            
        return {"brands": brands}
    except Exception as e:
        return {"error": str(e)}

@app.get("/map-stations")
def get_map_stations():
    """Returns all EV stations from the database for map display."""
    try:
        _db = SessionLocal()
        try:
            rows = _db.query(EVStation).filter(EVStation.is_active == True).all()
        finally:
            _db.close()

        stations_list = [{
            "stationID":  s.station_name or f"Station-{s.id}",
            "latitude":   s.latitude,
            "longitude":  s.longitude,
            "city":       s.city or "",
            "operator":   s.operator or "",
            "power":      "",
            "status":     "Available",
        } for s in rows if s.latitude and s.longitude]

        return {"stations": stations_list}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "EV ChargeSync API is running. Go to /docs for API documentation."}


# ── Station Clustering Endpoint ───────────────────────────────────────────────
@app.get("/stations/clusters")
def get_station_clusters():
    """
    Returns KMeans cluster assignments for all ACN stations.
    Each station is grouped by its 24-hour usage pattern (4 clusters).
    Response includes per-cluster average hourly demand profiles.
    """
    try:
        result = congestion_model.get_station_clusters()
        return result
    except Exception as e:
        return {"error": str(e)}

# ── Hybrid Forecast Endpoint ──────────────────────────────────────────────────
@app.get("/forecast/hybrid")
def forecast_hybrid():
    """
    Runs the ARIMA+LSTM hybrid forecasting pipeline on the ACN dataset.
    Returns a 24-hour station load forecast + peak hour prediction.
    Results are persisted to ForecastCache table (survives server restarts).
    """
    # Check DB cache first
    _db = SessionLocal()
    try:
        cached = _db.query(ForecastCache).order_by(ForecastCache.id.desc()).first()
        if cached:
            return cached.result
    finally:
        _db.close()

    try:
        from hybrid_forecast import run_pipeline
        HYBRID_DATA = r"d:\games\group11ziipp\group11\backend\acn_dataset.csv"
        result = run_pipeline(HYBRID_DATA)

        # Persist to DB
        _db2 = SessionLocal()
        try:
            _db2.add(ForecastCache(
                created_at=datetime.now(timezone.utc),
                result=result
            ))
            _db2.commit()
        finally:
            _db2.close()

        return result
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
