import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, Circle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

// ─── Static Icons ─────────────────────────────────────────────────────────────
const destIcon = L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
});

const sourceIcon = L.divIcon({
    className: 'source-icon',
    html: '<div style="position: relative; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;">' +
        '<div style="width: 16px; height: 16px; background-color: #2196F3; border: 3px solid white; border-radius: 50%; z-index: 10; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>' +
        '</div>',
    iconSize: [24, 24], iconAnchor: [12, 12], popupAnchor: [0, -12]
});

// ─── Pulsing highlighted station icon ───────────────────────────────────────
function makePulsingIcon() {
    return L.divIcon({
        className: '',
        html: `<div style="position:relative;width:40px;height:40px;display:flex;align-items:center;justify-content:center;">
            <div style="position:absolute;width:40px;height:40px;border-radius:50%;border:3px solid #22C55E;animation:pulse-ring 1.5s ease-out infinite;opacity:0.8;"></div>
            <div style="position:absolute;width:40px;height:40px;border-radius:50%;border:3px solid #22C55E;animation:pulse-ring 1.5s ease-out 0.5s infinite;opacity:0.5;"></div>
            <svg viewBox="0 0 24 36" width="26" height="38" style="filter:drop-shadow(0 0 6px #22C55E);">
                <path fill="#22C55E" stroke="white" stroke-width="2" d="M12 0C5.373 0 0 5.373 0 12c0 9 12 24 12 24s12-15 12-24c0-6.627-5.373-12-12-12z"/>
                <path fill="white" d="M11.5 7L6.5 15H11.5L9.5 21L16.5 13H11.5L13.5 7H11.5Z"/>
            </svg>
        </div>
        <style>@keyframes pulse-ring{0%{transform:scale(0.5);opacity:0.8}100%{transform:scale(1.8);opacity:0}}</style>`,
        iconSize: [40, 40],
        iconAnchor: [20, 38],
        popupAnchor: [0, -40]
    });
}

const pulsingIcon = makePulsingIcon();

const liveUserIcon = L.divIcon({
    className: 'custom-live-user-icon',
    html: '<div style="position: relative; width: 40px; height: 80px; display: flex; align-items: center; justify-content: center;">' +
        '<div style="position: absolute; width: 80px; height: 80px; background: radial-gradient(circle, rgba(33,150,243,0.4) 0%, rgba(33,150,243,0) 70%); pointer-events: none;"></div>' +
        '<img src="/ev-icon.png" style="width: 40px; height: 80px; object-fit: contain; z-index: 10; position: absolute;" />' +
        '</div>',
    iconSize: [40, 80], iconAnchor: [20, 60], popupAnchor: [0, -60]
});

// ─── Blue pin (all stations by default) ─────────────────────────────────────
function makeBlueIcon() {
    return L.divIcon({
        className: '',
        html: `<div style="position:relative;width:28px;height:42px;display:flex;justify-content:center;">
            <svg viewBox="0 0 24 36" width="28" height="42" style="filter:drop-shadow(0px 2px 3px rgba(0,0,0,0.35));">
                <path fill="#2196F3" stroke="white" stroke-width="2" d="M12 0C5.373 0 0 5.373 0 12c0 9 12 24 12 24s12-15 12-24c0-6.627-5.373-12-12-12z"/>
                <path fill="white" d="M11.5 7L6.5 15H11.5L9.5 21L16.5 13H11.5L13.5 7H11.5Z"/>
            </svg>
        </div>`,
        iconSize: [28, 42], iconAnchor: [14, 42], popupAnchor: [0, -42]
    });
}

// ─── Green pin (nearby stations ≤ 10 km from user) ───────────────────────────
function makeGreenIcon() {
    return L.divIcon({
        className: '',
        html: `<div style="position:relative;width:32px;height:48px;display:flex;justify-content:center;">
            <svg viewBox="0 0 24 36" width="32" height="48" style="filter:drop-shadow(0px 2px 5px rgba(34,197,94,0.6));">
                <path fill="#22C55E" stroke="white" stroke-width="2" d="M12 0C5.373 0 0 5.373 0 12c0 9 12 24 12 24s12-15 12-24c0-6.627-5.373-12-12-12z"/>
                <path fill="white" d="M11.5 7L6.5 15H11.5L9.5 21L16.5 13H11.5L13.5 7H11.5Z"/>
            </svg>
            <div style="position:absolute;top:-6px;right:-4px;background:#22C55E;color:white;border-radius:8px;font-size:8px;font-weight:900;padding:1px 4px;border:1.5px solid white;letter-spacing:0.5px;">NEAR</div>
        </div>`,
        iconSize: [32, 48], iconAnchor: [16, 48], popupAnchor: [0, -48]
    });
}

// ─── Suggested route station icon (amber) ────────────────────────────────────
function makeRouteStationIcon() {
    return L.divIcon({
        className: '',
        html: `<div style="position:relative;width:28px;height:42px;display:flex;justify-content:center;">
            <svg viewBox="0 0 24 36" width="28" height="42" style="filter:drop-shadow(0px 2px 3px rgba(0,0,0,0.35));">
                <path fill="#F59E0B" stroke="white" stroke-width="2" d="M12 0C5.373 0 0 5.373 0 12c0 9 12 24 12 24s12-15 12-24c0-6.627-5.373-12-12-12z"/>
                <path fill="white" d="M11.5 7L6.5 15H11.5L9.5 21L16.5 13H11.5L13.5 7H11.5Z"/>
            </svg>
        </div>`,
        iconSize: [28, 42], iconAnchor: [14, 42], popupAnchor: [0, -42]
    });
}

const blueIcon = makeBlueIcon();
const greenIcon = makeGreenIcon();
const routeEvIcon = makeRouteStationIcon();

// ─── Haversine distance helper ────────────────────────────────────────────────
function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ─── Distance Badge Overlay (shown when trip route is active) ─────────────────
function RouteDistanceBadge({ route }) {
    if (!route || route.length < 2) return null;
    // Sum polyline segment lengths
    let totalKm = 0;
    for (let i = 1; i < route.length; i++) {
        totalKm += haversineKm(route[i - 1][0], route[i - 1][1], route[i][0], route[i][1]);
    }
    const km = totalKm.toFixed(1);
    const mins = Math.round(totalKm * 1.8); // rough drive estimate
    return (
        <div style={{
            position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
            zIndex: 1000, background: 'rgba(2,8,23,0.88)', backdropFilter: 'blur(8px)',
            border: '1px solid rgba(34,211,238,0.35)', borderRadius: 20,
            padding: '6px 18px', display: 'flex', alignItems: 'center', gap: 10,
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)'
        }}>
            <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1 }}>Route</span>
            <span style={{ fontSize: 16, color: '#22d3ee', fontWeight: 900 }}>{km} km</span>
            <span style={{ color: '#475569', fontSize: 13 }}>·</span>
            <span style={{ fontSize: 13, color: '#a5f3fc', fontWeight: 700 }}>~{mins} min</span>
        </div>
    );
}

// ─── Congestion popup inner content ──────────────────────────────────────────
function CongestionPopup({ station, cong }) {
    const level = cong?.level || 'Low';
    const pct = cong?.congestion_pct ?? 0;
    const wait = cong?.predicted_wait_minutes ?? 0;
    const available = cong?.available_chargers ?? '?';
    const total = cong?.total_chargers ?? '?';
    const spark = cong?.sparkline_6h || [];
    const levelColor = { Low: '#22C55E', Moderate: '#F59E0B', Busy: '#EF4444' };
    const barColor = levelColor[level] || '#22C55E';

    return (
        <div style={{ fontFamily: 'sans-serif', minWidth: 200, fontSize: 13, lineHeight: 1.5 }}>
            <b style={{ fontSize: 14 }}>{station.stationID || 'Charging Station'}</b>
            {station.city && <div style={{ color: '#888', fontSize: 11 }}>{station.city}</div>}

            <div style={{ marginTop: 8, background: '#111', borderRadius: 8, padding: '8px 10px', color: '#fff' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ color: barColor, fontWeight: 'bold', fontSize: 13 }}>⚡ {level} Congestion</span>
                    <span style={{ fontSize: 11, color: '#ccc' }}>{available}/{total} free</span>
                </div>

                {/* Congestion bar */}
                <div style={{ background: '#333', borderRadius: 4, height: 6, marginBottom: 6, overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: barColor, borderRadius: 4, transition: 'width 0.4s' }} />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#aaa' }}>
                    <span>Occupancy: <b style={{ color: '#fff' }}>{pct}%</b></span>
                    <span>Wait: <b style={{ color: wait > 0 ? '#EF4444' : '#22C55E' }}>{wait > 0 ? `~${wait} min` : 'None'}</b></span>
                </div>

                {/* Sparkline */}
                {spark.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                        <div style={{ fontSize: 10, color: '#888', marginBottom: 3 }}>6-Hour Forecast</div>
                        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 32 }}>
                            {spark.map((v, i) => {
                                const h = Math.max(4, Math.round((v / 100) * 32));
                                const c = v < 40 ? '#22C55E' : v < 70 ? '#F59E0B' : '#EF4444';
                                return <div key={i} style={{ flex: 1, height: h, background: c, borderRadius: 2, opacity: 0.85 }} title={`+${i}h: ${v}%`} />;
                            })}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#666', marginTop: 2 }}>
                            <span>Now</span><span>+6h</span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// ─── Route Bounds ─────────────────────────────────────────────────────────────
const RouteBounds = ({ route, userLocation }) => {
    const map = useMap();
    useEffect(() => {
        if (route && route.length > 0) {
            const bounds = L.latLngBounds(route);
            map.fitBounds(bounds, { padding: [50, 50] });
        } else if (userLocation) {
            map.setView(userLocation, 12);
        }
    }, [route, userLocation, map]);
    return null;
};

// ─── Main Map Component ───────────────────────────────────────────────────────
const MapComponent = ({ userLocation, setUserLocation, stations, route, predictedStations, smartRoute, highlightedStation }) => {
    const center = userLocation || [20.5937, 78.9629];
    const initialZoom = userLocation ? 13 : 5;
    const startPoint = (route && route.length > 0) ? route[0] : userLocation;

    // Compute which station coords are "nearby" (≤10 km from user)
    const nearbySet = new Set();
    if (userLocation) {
        (stations || []).forEach((st, idx) => {
            const dist = haversineKm(userLocation[0], userLocation[1], st.latitude || 0, st.longitude || 0);
            if (dist <= 10) nearbySet.add(idx);
        });
    }

    // Fetch congestion for all visible ACN stations (batch call once)
    const [congestionMap, setCongestionMap] = useState({});
    const fetchedRef = useRef(false);

    useEffect(() => {
        if (fetchedRef.current) return;
        fetchedRef.current = true;
        axios.get(`${API_BASE}/stations/congestion`)
            .then(res => {
                if (res.data?.congestion) {
                    const map = {};
                    res.data.congestion.forEach(s => { map[s.station_id] = s; });
                    setCongestionMap(map);
                }
            })
            .catch(() => { });
    }, []);

    return (
        <div className="h-[600px] w-full rounded-3xl overflow-hidden border border-slate-800/50 relative z-0">
            {/* Route distance badge overlay */}
            <RouteDistanceBadge route={route} />

            {/* Map legend */}
            <div style={{
                position: 'absolute', bottom: 10, left: 10, zIndex: 1000,
                background: 'rgba(2,8,23,0.82)', backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12,
                padding: '6px 12px', display: 'flex', flexDirection: 'column', gap: 4
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: '#94a3b8', fontWeight: 700 }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#22C55E', border: '1.5px solid white' }} />
                    Nearby Station (&le;10 km)
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: '#94a3b8', fontWeight: 700 }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#2196F3', border: '1.5px solid white' }} />
                    Charging Station
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: '#94a3b8', fontWeight: 700 }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#F59E0B', border: '1.5px solid white' }} />
                    Route Stop
                </div>
            </div>
            <MapContainer center={center} zoom={initialZoom} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                <RouteBounds route={route} userLocation={userLocation} />

                {/* User GPS Marker */}
                {userLocation ? (
                    <React.Fragment>
                        <Marker
                            position={userLocation}
                            icon={liveUserIcon}
                            draggable={true}
                            eventHandlers={{
                                dragend: (e) => {
                                    const pos = e.target.getLatLng();
                                    if (setUserLocation) setUserLocation([pos.lat, pos.lng]);
                                }
                            }}
                        >
                            <Popup>Your GPS Location (Drag to adjust)</Popup>
                        </Marker>
                        <Circle center={userLocation} radius={150} pathOptions={{ color: '#2196F3', fillColor: '#2196F3', fillOpacity: 0.2, weight: 1 }} />
                    </React.Fragment>
                ) : null}

                {/* Trip start point */}
                {(route && route.length > 0 && startPoint) ? (
                    <Marker position={startPoint} icon={sourceIcon}>
                        <Popup>Start of Trip</Popup>
                    </Marker>
                ) : null}

                {/* All station markers — blue for all, green for nearby */}
                {stations ? stations.map((station, idx) => {
                    const cong = congestionMap[station.stationID];
                    const isNearby = nearbySet.has(idx);
                    const icon = isNearby ? greenIcon : blueIcon;
                    return (
                        <Marker key={idx} position={[station.latitude || 20.5937, station.longitude || 78.9629]} icon={icon}>
                            <Popup>
                                <CongestionPopup station={station} cong={cong} />
                            </Popup>
                        </Marker>
                    );
                }) : null}

                {/* Predicted stations along the route */}
                {predictedStations ? predictedStations.map((station, idx) => {
                    const icon = routeEvIcon;
                    const congProxy = {
                        level: station.congestion_level || 'Low',
                        congestion_pct: station.congestion_pct || 0,
                        predicted_wait_minutes: station.predicted_wait_minutes || 0,
                        available_chargers: station.available_chargers,
                        total_chargers: station.total_chargers,
                    };
                    return (
                        <Marker key={`pred-${idx}`} position={[station.latitude, station.longitude]} icon={icon}>
                            <Popup>
                                <CongestionPopup station={{ stationID: station.stationID, city: station.city }} cong={congProxy} />
                            </Popup>
                        </Marker>
                    );
                }) : null}

                {/* Destination */}
                {(route && route.length > 0) ? (
                    <Marker position={route[route.length - 1]} icon={destIcon}>
                        <Popup>Destination</Popup>
                    </Marker>
                ) : null}

                {/* Smart route polyline — user to selected station */}
                {smartRoute && smartRoute.length > 1 ? (
                    <Polyline
                        positions={smartRoute}
                        color="#22C55E"
                        weight={4}
                        opacity={0.85}
                        dashArray="10, 6"
                    />
                ) : null}

                {/* Glowing pulsing marker for selected station */}
                {highlightedStation ? (
                    <Marker position={highlightedStation} icon={pulsingIcon}>
                        <Popup>Selected Charging Station</Popup>
                    </Marker>
                ) : null}

                {route ? <Polyline positions={route} color="#22d3ee" weight={5} opacity={0.85} /> : null}
            </MapContainer>
        </div>
    );
};

// ─── Error Boundary ───────────────────────────────────────────────────────────
class MapErrorBoundary extends React.Component {
    constructor(props) { super(props); this.state = { hasError: false, error: null, errorInfo: null }; }
    static getDerivedStateFromError(error) { return { hasError: true }; }
    componentDidCatch(error, errorInfo) { this.setState({ error, errorInfo }); }
    render() {
        if (this.state.hasError) {
            return (
                <div style={{ padding: '20px', backgroundColor: '#fee2e2', color: '#991b1b', borderRadius: '1rem', height: '600px', overflow: 'auto' }}>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>Map Render Error</h2>
                    <p style={{ marginTop: '10px', fontWeight: 'bold' }}>{this.state.error && this.state.error.toString()}</p>
                    <pre style={{ marginTop: '10px', fontSize: '12px', whiteSpace: 'pre-wrap' }}>
                        {this.state.errorInfo && this.state.errorInfo.componentStack}
                    </pre>
                </div>
            );
        }
        return this.props.children;
    }
}

const WrappedMapComponent = (props) => (
    <MapErrorBoundary>
        <MapComponent {...props} />
    </MapErrorBoundary>
);

export default WrappedMapComponent;
