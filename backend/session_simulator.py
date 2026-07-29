"""
session_simulator.py

Real-Time EV Charging Session Simulator (Method 2 + Method 3 Hybrid)

Method 2: LSTM-predicted hourly sessions → Poisson arrival rate → probabilistic 
          new vehicle arrivals each tick.
Method 3: Each simulated session has a start_time + random charging_duration; 
          auto-expires when time elapses.

Usage:
    from session_simulator import SessionSimulator
    sim = SessionSimulator(congestion_model)
    await sim.start()          # call once inside FastAPI lifespan/startup
    state = sim.get_state("2-39-139-28")
"""

import asyncio
import time
import random
import math
from datetime import datetime, timezone
from typing import Dict, List


# ── Default charger count if we can't infer it ──────────────────────────────
DEFAULT_CHARGERS = 4
TICK_SECS = 10          # How often the simulator updates
MIN_SESSION_MINS = 20   # Minimum charging duration (minutes)
MAX_SESSION_MINS = 90   # Maximum charging duration (minutes)


class Session:
    """Represents one active charging session."""
    __slots__ = ("start_time", "duration_secs")

    def __init__(self):
        self.start_time = time.time()
        self.duration_secs = random.randint(MIN_SESSION_MINS * 60, MAX_SESSION_MINS * 60)

    @property
    def remaining_secs(self) -> float:
        return max(0.0, self.start_time + self.duration_secs - time.time())

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.start_time + self.duration_secs


class StationState:
    """Live state for one charging station."""

    def __init__(self, station_id: str, total_chargers: int):
        self.station_id = station_id
        self.total_chargers = max(1, total_chargers)
        self.sessions: List[Session] = []

    # Seed with realistic initial occupancy based on current-hour profile
    def seed(self, avg_sessions_now: float):
        count = min(round(avg_sessions_now), self.total_chargers)
        for _ in range(count):
            s = Session()
            # Randomise start so sessions don't all expire at the same time
            s.start_time = time.time() - random.randint(0, MAX_SESSION_MINS * 60)
            self.sessions.append(s)

    def tick(self, arrival_prob: float):
        """Remove expired sessions and probabilistically add new ones."""
        # Method 3 — expire finished sessions
        self.sessions = [s for s in self.sessions if not s.is_expired]

        # Method 2 — Poisson-driven arrival
        active = len(self.sessions)
        if active < self.total_chargers and random.random() < arrival_prob:
            self.sessions.append(Session())

    # ── Derived metrics ──────────────────────────────────────────────────────
    @property
    def active_count(self) -> int:
        return len(self.sessions)

    @property
    def available(self) -> int:
        return max(0, self.total_chargers - self.active_count)

    @property
    def occupancy_pct(self) -> int:
        return int((self.active_count / self.total_chargers) * 100)

    @property
    def congestion_level(self) -> str:
        p = self.occupancy_pct
        if p < 40:
            return "Low"
        elif p < 70:
            return "Moderate"
        return "Busy"

    @property
    def predicted_wait_minutes(self) -> int:
        if self.available > 0:
            return 0
        # Estimate wait as average remaining session time of the shortest session
        if not self.sessions:
            return 0
        shortest_remaining = min(s.remaining_secs for s in self.sessions)
        return int(shortest_remaining / 60)

    def as_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "total_chargers": self.total_chargers,
            "active_sessions": self.active_count,
            "available_chargers": self.available,
            "occupancy_pct": self.occupancy_pct,
            "congestion_pct": self.occupancy_pct,
            "level": self.congestion_level,
            "congestion_level": self.congestion_level,
            "predicted_wait_minutes": self.predicted_wait_minutes,
        }


# ── Main simulator class ─────────────────────────────────────────────────────

class SessionSimulator:
    """
    Manages live charging session state for all ACN stations.
    Uses CongestionModel to get hourly profiles for Poisson arrivals.
    """

    def __init__(self, congestion_model):
        self.model = congestion_model
        self._states: Dict[str, StationState] = {}
        self._running = False
        self._task = None

    # ── Initialise station states ─────────────────────────────────────────
    def _init_states(self):
        if self.model.df is None:
            return

        station_ids = self.model.df["stationID"].unique().tolist()
        hour_now = datetime.now(timezone.utc).hour
        dow_now = datetime.now(timezone.utc).weekday()

        for sid in station_ids:
            # Estimate total chargers
            st_df = self.model.df[self.model.df["stationID"] == sid]
            days = max(1, len(self.model.df["connectionTime"].dt.date.unique()))
            total = max(2, int(len(st_df) / days))
            total = min(total, 10)  # Cap at 10 to be realistic

            state = StationState(sid, total)

            # Seed with current-hour average occupancy
            profile = self.model.station_profiles.get(sid, {})
            avg_now = profile.get((hour_now, dow_now), 1.0)
            state.seed(avg_now)
            self._states[sid] = state

        print(f"[SessionSimulator] Initialised {len(self._states)} stations OK")

    # ── Tick: called every TICK_SECS ─────────────────────────────────────
    def _do_tick(self):
        hour_now = datetime.now(timezone.utc).hour
        dow_now = datetime.now(timezone.utc).weekday()

        for sid, state in self._states.items():
            profile = self.model.station_profiles.get(sid, {})
            predicted_per_hour = profile.get((hour_now, dow_now), 1.0)

            # Convert predicted sessions/hour → P(arrival in TICK_SECS window)
            # Using Poisson: lambda = rate * time_window
            lam = (predicted_per_hour / 3600) * TICK_SECS
            # P(≥1 arrival) = 1 - e^(-lambda)
            arrival_prob = 1 - math.exp(-lam)

            state.tick(arrival_prob)

    # ── Asyncio background loop ───────────────────────────────────────────
    async def _run_loop(self):
        self._init_states()
        while self._running:
            await asyncio.sleep(TICK_SECS)
            self._do_tick()

    def start(self):
        """Start the background simulation loop (non-blocking)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        print("[SessionSimulator] Background loop started OK")

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    # ── Public API ────────────────────────────────────────────────────────
    def get_state(self, station_id: str) -> dict:
        """Return live state dict for one station."""
        if station_id in self._states:
            return self._states[station_id].as_dict()
        # Unknown station — fall back to congestion model prediction
        pred = self.model.predict(station_id, hours_ahead=0)
        return {
            **pred,
            "occupancy_pct": pred.get("congestion_pct", 0),
            "active_sessions": pred.get("predicted_active_sessions", 0),
        }

    def get_all_states(self) -> list:
        """Return live state dicts for all known stations."""
        return [s.as_dict() for s in self._states.values()]

    def get_enriched_state(self, station_id: str) -> dict:
        """Live state + 6-hour sparkline from the LSTM profile."""
        base = self.get_state(station_id)
        # Append sparkline from congestion model (profile-based, fast)
        pred = self.model.predict(station_id, hours_ahead=0)
        base["sparkline_6h"] = pred.get("sparkline_6h", [base["congestion_pct"]] * 6)
        return base
