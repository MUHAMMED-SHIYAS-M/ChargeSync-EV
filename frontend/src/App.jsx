import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Zap,
  MapPin,
  Clock,
  TrendingUp,
  Battery,
  Search,
  Navigation,
  Info,
  CheckCircle2,
  AlertCircle,
  Activity,
  Layers,
  Settings,
  ChevronRight,
  Loader
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import MapComponent from './components/MapComponent';

const API_BASE = 'http://localhost:8000';

function App() {
  const [stations, setStations] = useState([]);
  const [mapStations, setMapStations] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [batteryLevel, setBatteryLevel] = useState(25);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [stats, setStats] = useState({
    totalNodes: 0,
    avgWaitMinutes: 0,
    totalMegawatts: 0,
    peakLoad: 0
  });
  // Vehicle Selection State
  const [vehicleData, setVehicleData] = useState({});
  const [selectedEV, setSelectedEV] = useState('');
  const [vehicleStats, setVehicleStats] = useState(null);

  const [tripPlan, setTripPlan] = useState({
    source: '',
    destination: '',
    route: null
  });
  const [userLocation, setUserLocation] = useState(null);
  const [congestionForecast, setCongestionForecast] = useState({ labels: [], congestion_pct: [] });
  const [optimizeForWait, setOptimizeForWait] = useState(false);
  const [vehicleRange, setVehicleRange] = useState(400); // km, synced from EV selector
  const [smartRoutes, setSmartRoutes] = useState([]);
  const [selectedRouteIdx, setSelectedRouteIdx] = useState(null);
  const [smartRoutePolyline, setSmartRoutePolyline] = useState(null);
  const [highlightedStation, setHighlightedStation] = useState(null);
  const [smartRoutesLoading, setSmartRoutesLoading] = useState(false);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const res = await axios.get(`${API_BASE}/stations`);
      setStations(res.data.stations);

      try {
        const mapRes = await axios.get(`${API_BASE}/map-stations`);
        if (mapRes.data && mapRes.data.stations) {
          setMapStations(mapRes.data.stations);
        }
      } catch (err) {
        console.error("Map API Error:", err);
      }

      const predMap = {};
      for (const id of res.data.stations) {
        const p = await axios.get(`${API_BASE}/predict?station_id=${id}`);
        predMap[id] = p.data;
      }
      setPredictions(predMap);

      const opt = await axios.get(`${API_BASE}/optimize?battery=${batteryLevel}`);
      setRecommendations(opt.data);

      const s = await axios.get(`${API_BASE}/stats`);
      setStats(s.data);

      const v = await axios.get(`${API_BASE}/vehicles`);
      if (v.data && v.data.brands) {
        setVehicleData(v.data.brands);
      }

      // Fetch congestion forecast for dashboard
      try {
        const cf = await axios.get(`${API_BASE}/stations/congestion/forecast`);
        if (cf.data && cf.data.labels) setCongestionForecast(cf.data);
      } catch (_) { }

      // Check if user is on Map tab and trigger geolocation if no userLocation yet
      if (activeTab === 'map' && !userLocation) {
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (pos) => {
              setUserLocation([pos.coords.latitude, pos.coords.longitude]);
            },
            (err) => {
              console.warn("Geolocation permission error: ", err.message);
            },
            { enableHighAccuracy: true, maximumAge: 0, timeout: 20000 }
          );
        }
      }

      setLoading(false);
    } catch (err) {
      console.error("API Error:", err);
      // Fallback data
      setStations(['CA-303', 'CA-309', 'CA-305', 'CA-313', 'CA-315']);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [activeTab]);

  // Fetch smart routes whenever user location changes (GPS granted or pin dragged)
  useEffect(() => {
    if (!userLocation) return;
    const [lat, lng] = userLocation;
    const fetchSmartRoutes = async () => {
      try {
        setSmartRoutesLoading(true);
        const res = await axios.get(`${API_BASE}/stations/smart-routes?lat=${lat}&lng=${lng}&radius=10&limit=8`);
        if (res.data?.routes) {
          setSmartRoutes(res.data.routes);
          setSelectedRouteIdx(null);
          setSmartRoutePolyline(null);
          setHighlightedStation(null);
        }
      } catch (e) {
        console.error('Smart routes fetch failed', e);
      } finally {
        setSmartRoutesLoading(false);
      }
    };
    fetchSmartRoutes();
    const interval = setInterval(fetchSmartRoutes, 30000);
    return () => clearInterval(interval);
  }, [userLocation]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setSearching(true);
    try {
      // Default location (e.g., Los Angeles) if geolocation is not used
      const lat = 34.0522;
      const lng = -118.2437;

      const res = await axios.get(`${API_BASE}/places/nearby?lat=${lat}&lng=${lng}&radius=5000&type=charging_station&keyword=${searchQuery}`);
      if (res.data.results) {
        setSearchResults(res.data.results);
        console.log("Search results:", res.data.results);
      }
    } catch (err) {
      console.error("Search error:", err);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] flex">
      {/* Sidebar */}
      <aside className="w-72 bg-slate-950/50 border-r border-slate-800/50 backdrop-blur-2xl p-8 flex flex-col fixed h-full z-20">
        <div className="flex items-center gap-4 mb-12">
          <div className="w-12 h-12 bg-cyan-400 rounded-2xl flex items-center justify-center shadow-[0_0_20px_rgba(34,211,238,0.4)]">
            <Zap className="text-slate-950" size={24} strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight font-display bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">ChargeSync</h1>
            <p className="text-[10px] uppercase tracking-widest text-cyan-400 font-bold">Smart Flow v2.0</p>
          </div>
        </div>

        <nav className="flex flex-col gap-3">
          <NavItem icon={<Activity size={20} />} label="Network Status" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
          <NavItem icon={<Navigation size={20} />} label="Trip Optimizer" active={activeTab === 'optimizer'} onClick={() => setActiveTab('optimizer')} />
          <NavItem icon={<Layers size={20} />} label="Station Map" active={activeTab === 'map'} onClick={() => setActiveTab('map')} />
          <NavItem icon={<Settings size={20} />} label="Settings" active={activeTab === 'settings'} onClick={() => setActiveTab('settings')} />
        </nav>

        <div className="mt-auto space-y-6">
          <div className="p-6 bg-slate-900/40 rounded-3xl border border-slate-800/50 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-400/5 rounded-full -mr-8 -mt-8 blur-2xl group-hover:bg-cyan-400/10 transition-all" />
            <div className="flex justify-between items-center mb-3 relative z-10">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Battery Health</span>
              <span className={`text-sm font-black ${batteryLevel < 20 ? 'text-rose-500' : 'text-cyan-400'}`}>{batteryLevel}%</span>
            </div>
            <div className="w-full bg-slate-800/80 h-1.5 rounded-full overflow-hidden relative z-10">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${batteryLevel}%` }}
                className={`h-full ${batteryLevel < 20 ? 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]' : 'bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]'}`}
              />
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 ml-72 p-10 min-h-screen relative">
        <div className="absolute top-0 right-0 p-10 w-full h-[500px] bg-gradient-to-b from-cyan-500/5 to-transparent pointer-events-none -z-10" />

        {activeTab === 'dashboard' && (
          <header className="flex justify-between items-end mb-12">
            <div className="space-y-2">
              <h2 className="text-4xl font-black tracking-tight font-display text-white">System Overview</h2>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-[11px] font-black text-emerald-400 uppercase tracking-widest">Real-time Feed</span>
                </div>
                <span className="text-slate-500 text-sm">Last updated: 1 min ago</span>
              </div>
            </div>

            <div className="flex flex-col gap-2 items-end">
              {/* EV Chooser Dropdowns */}
              <div className="flex flex-col items-end gap-2 bg-slate-900/50 p-3 rounded-2xl border border-slate-800">
                <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase tracking-widest w-full px-1">
                  <Zap size={14} className="text-cyan-400" /> Vehicle Selector
                  {vehicleStats && (
                    <span className="ml-auto text-emerald-400 bg-emerald-900/30 px-2 py-0.5 rounded-md">
                      {vehicleStats.range}km Range
                    </span>
                  )}
                </div>
                <div className="flex gap-2 w-full">
                  <select
                    className="flex-1 bg-slate-950 border border-slate-700 text-white text-sm rounded-xl px-4 py-2 outline-none focus:border-cyan-400 min-w-[250px]"
                    value={selectedEV}
                    onChange={(e) => {
                      const val = e.target.value;
                      setSelectedEV(val);
                      if (!val) {
                        setVehicleStats(null);
                        return;
                      }

                      const [brand, ...rest] = val.split(' - ');
                      const model = rest.join(' - ');

                      if (vehicleData[brand]) {
                        const stats = vehicleData[brand].find(m => m.model === model);
                        setVehicleStats(stats || null);
                        // Sync vehicle range for battery-aware trip planning
                        if (stats && stats.range && !isNaN(parseFloat(stats.range))) {
                          setVehicleRange(parseFloat(stats.range));
                        }
                      }
                    }}
                  >
                    <option value="">Select EV...</option>
                    {Object.keys(vehicleData).map(b => (
                      <optgroup key={b} label={b}>
                        {vehicleData[b].map(m => (
                          <option key={`${b} - ${m.model}`} value={`${b} - ${m.model}`}>
                            {b} {m.model}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </div>
              </div>

              <button className="px-6 py-3 bg-white text-slate-950 rounded-2xl font-bold flex items-center justify-center gap-3 hover:bg-slate-200 transition-all shadow-xl hover:shadow-cyan-400/10 transform hover:-translate-y-1 w-full">
                <Navigation size={18} fill="currentColor" />
                Start Smart Trip
              </button>
            </div>
          </header>
        )}

        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' ? (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.4, ease: "circOut" }}
              className="grid grid-cols-12 gap-8"
            >
              {/* Main Stats */}
              <div className="col-span-12 grid grid-cols-4 gap-6">
                <StatCard title="Active Nodes" value={stats.totalNodes || stations.length} sub="Real-time stations" icon={<TrendingUp size={20} />} color="cyan" />
                <StatCard title="Global Wait" value={`${stats.avgWaitMinutes}m`} sub="Average wait time" icon={<Clock size={20} />} color="violet" />
                <StatCard title="Total Capacity" value={stats.totalMegawatts} sub="Megawatts online" icon={<Zap size={20} />} color="amber" />
                <StatCard title="Peak Demand" value={stats.peakLoad} sub="Max concurrent load" icon={<Activity size={20} />} color="emerald" />
              </div>

              {/* Station Control Center */}
              <div className="col-span-8 space-y-8">
                <div className="bg-slate-900/30 border border-slate-800/50 rounded-[40px] p-10 backdrop-blur-3xl glow relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-8">
                    <form onSubmit={handleSearch} className="relative group">
                      <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-cyan-400 transition-colors" size={18} />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search places..."
                        className="bg-slate-950/50 rounded-2xl pl-12 pr-6 py-3 text-sm border border-slate-800 outline-none focus:border-cyan-400/50 focus:bg-slate-950 transition-all w-64"
                      />
                    </form>
                  </div>

                  <h3 className="text-2xl font-bold mb-10 text-white font-display">Live Station Feed</h3>

                  <div className="space-y-4">
                    {/* Display Search Results if available */}
                    {searchResults.length > 0 && (
                      <div className="mb-6 p-4 bg-cyan-900/20 border border-cyan-500/30 rounded-2xl">
                        <h4 className="text-cyan-400 font-bold mb-2">Google Places Results ({searchResults.length})</h4>
                        {searchResults.slice(0, 3).map((place) => (
                          <div key={place.place_id} className="mb-2 p-2 bg-slate-900/50 rounded-lg">
                            <p className="text-white font-bold">{place.name}</p>
                            <p className="text-slate-400 text-xs">{place.vicinity}</p>
                          </div>
                        ))}
                      </div>
                    )}
                    {stations.map((id, idx) => (
                      <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        key={id}
                        className="group flex items-center justify-between p-6 bg-slate-900/20 hover:bg-slate-900/50 border border-slate-800/40 hover:border-slate-700/50 rounded-3xl transition-all cursor-pointer relative overflow-hidden"
                      >
                        <div className="absolute left-0 top-0 w-1 h-full bg-transparent group-hover:bg-cyan-400 transition-all" />
                        <div className="flex items-center gap-6 relative z-10">
                          <div className="w-14 h-14 bg-slate-950 rounded-2xl flex items-center justify-center text-slate-400 group-hover:text-cyan-400 border border-slate-800 transition-all">
                            <MapPin size={24} />
                          </div>
                          <div>
                            <p className="text-lg font-bold text-white mb-0.5">{id}</p>
                            <div className="flex items-center gap-3">
                              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Zone A-4</span>
                              <div className="w-1 h-1 rounded-full bg-slate-700" />
                              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">3.5km away</span>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-12 relative z-10">
                          <div className="text-right">
                            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Status</p>
                            <span className={`flex items-center gap-2 font-bold ${predictions[id]?.status === 'Busy' ? 'text-rose-400' : 'text-emerald-400'}`}>
                              <span className={`w-1.5 h-1.5 rounded-full ${predictions[id]?.status === 'Busy' ? 'bg-rose-400' : 'bg-emerald-400'}`} />
                              {predictions[id]?.status || 'Idle'}
                            </span>
                          </div>
                          <div className="text-right min-w-[80px]">
                            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Est. Wait</p>
                            <p className="font-bold text-white">{predictions[id]?.predictedWaitMinutes || '--'}m</p>
                          </div>
                          <ChevronRight className="text-slate-700 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" size={20} />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Sidebar Cards */}
              <div className="col-span-4 space-y-8">
                <div className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-[40px] p-8 shadow-2xl shadow-cyan-500/20 relative overflow-hidden group">
                  <div className="absolute bottom-0 right-0 w-48 h-48 bg-white/10 rounded-full -mb-24 -mr-24 blur-3xl group-hover:bg-white/20 transition-all" />
                  <div className="relative z-10 h-full flex flex-col">
                    <div className="flex items-center gap-3 mb-8">
                      <div className="w-10 h-10 bg-white/20 backdrop-blur-lg rounded-xl flex items-center justify-center">
                        <Navigation className="text-white" size={20} fill="currentColor" />
                      </div>
                      <h3 className="text-xl font-black text-white font-display">Smart Route</h3>
                    </div>

                    {recommendations.length > 0 ? (
                      <div className="flex-1 space-y-6">
                        <div className="space-y-1">
                          <p className="text-cyan-100/70 text-[10px] font-bold uppercase tracking-widest">Recommended Stop</p>
                          <p className="text-3xl font-black text-white">{recommendations[0].stationID}</p>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-white/10 backdrop-blur-xl rounded-2xl border border-white/10">
                            <p className="text-[10px] font-bold text-cyan-100 uppercase tracking-widest mb-1">Time Saved</p>
                            <p className="text-xl font-black text-white">12.5m</p>
                          </div>
                          <div className="p-4 bg-white/10 backdrop-blur-xl rounded-2xl border border-white/10">
                            <p className="text-[10px] font-bold text-cyan-100 uppercase tracking-widest mb-1">Queue Pos.</p>
                            <p className="text-xl font-black text-white">Next</p>
                          </div>
                        </div>

                        <button className="w-full py-5 bg-white text-blue-600 rounded-3xl font-black text-sm uppercase tracking-widest hover:bg-slate-100 transition-all transform hover:scale-[1.02] shadow-xl">
                          Apply Optimization
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center py-12 gap-4">
                        <div className="w-12 h-12 border-4 border-cyan-200/30 border-t-white rounded-full animate-spin" />
                        <p className="text-white font-bold animate-pulse text-sm">Parsing ACN Data...</p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="bg-slate-900/30 border border-slate-800/50 rounded-[40px] p-8 backdrop-blur-3xl">
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="text-xl font-bold text-white font-display">System Load</h3>
                    <Activity className="text-cyan-400" size={18} />
                  </div>
                  <div className="space-y-6">
                    <LoadItem label="Processing Nodes" percent={78} color="cyan" />
                    <LoadItem label="Data Throughput" percent={45} color="emerald" />
                    <LoadItem label="API Latency" percent={12} color="amber" />
                  </div>
                </div>

                {/* Congestion Forecast Panel */}
                <div className="bg-slate-900/30 border border-slate-800/50 rounded-[40px] p-8 backdrop-blur-3xl">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-white font-display">Congestion Forecast</h3>
                    <span className="text-xs text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 px-2 py-1 rounded-full">Next 6h</span>
                  </div>
                  {congestionForecast.labels.length > 0 ? (
                    <div className="space-y-3">
                      {congestionForecast.labels.map((label, i) => {
                        const pct = congestionForecast.congestion_pct[i] || 0;
                        const barColor = pct < 40 ? 'bg-emerald-500' : pct < 70 ? 'bg-amber-400' : 'bg-rose-500';
                        const textColor = pct < 40 ? 'text-emerald-400' : pct < 70 ? 'text-amber-400' : 'text-rose-400';
                        return (
                          <div key={i} className="space-y-1">
                            <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest">
                              <span className="text-slate-400">{label}</span>
                              <span className={textColor}>{pct}%</span>
                            </div>
                            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                              <div className={`h-full ${barColor} transition-all duration-700`} style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-sm text-center py-4">Loading forecast...</p>
                  )}
                </div>
              </div>
            </motion.div>
          ) : activeTab === 'map' ? (
            <div className="h-full w-full p-4 flex flex-col gap-6">
              <MapComponent
                userLocation={userLocation || (
                  tripPlan.source &&
                    tripPlan.source.includes(',') &&
                    tripPlan.source.split(',').every(n => !isNaN(parseFloat(n.trim())))
                    ? tripPlan.source.split(',').map(n => parseFloat(n.trim()))
                    : null
                )}
                setUserLocation={setUserLocation}
                stations={mapStations.length > 0 ? mapStations : stations.map(id => ({ stationID: id, latitude: 34.1 + Math.random() * 0.1, longitude: -118.2 + Math.random() * 0.1, status: 'Available' }))}
                route={tripPlan.route}
                predictedStations={tripPlan.suggested}
                smartRoute={smartRoutePolyline}
                highlightedStation={highlightedStation}
              />

              {/* Smart Route Recommendations */}
              {userLocation && (
                <SmartRouteList
                  routes={smartRoutes}
                  loading={smartRoutesLoading}
                  selectedIdx={selectedRouteIdx}
                  onSelect={(idx, route) => {
                    setSelectedRouteIdx(idx);
                    setSmartRoutePolyline(route.route_polyline);
                    setHighlightedStation([route.latitude, route.longitude]);
                  }}
                />
              )}
            </div>
          ) : activeTab === 'optimizer' ? (
            <Optimizer
              recommendations={recommendations}
              batteryLevel={batteryLevel}
              setBatteryLevel={setBatteryLevel}
              vehicleRange={vehicleRange}
              setVehicleRange={setVehicleRange}
              vehicleData={vehicleData}
              selectedEV={selectedEV}
              setSelectedEV={setSelectedEV}
              vehicleStats={vehicleStats}
              setVehicleStats={setVehicleStats}
              onRefresh={fetchData}
              setActiveTab={setActiveTab}
              tripPlan={tripPlan}
              setTripPlan={setTripPlan}
              optimizeForWait={optimizeForWait}
              setOptimizeForWait={setOptimizeForWait}
            />
          ) : activeTab === 'settings' ? (
            <div className="flex items-center justify-center h-64">
              <p className="text-slate-500 text-lg font-bold">Settings coming soon...</p>
            </div>
          ) : null}
        </AnimatePresence>
      </main>
    </div>
  );
}

function Optimizer({ recommendations, batteryLevel, setBatteryLevel, vehicleRange = 400, setVehicleRange, vehicleData = {}, selectedEV = '', setSelectedEV, vehicleStats, setVehicleStats, onRefresh, setActiveTab, tripPlan, setTripPlan, optimizeForWait, setOptimizeForWait }) {
  const reachableKm = tripPlan.reachableKm ?? Math.round((batteryLevel / 100) * vehicleRange);
  return (
    <motion.div
      key="optimizer"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.4 }}
      className="max-w-4xl mx-auto space-y-12 pb-20"
    >
      <div className="bg-slate-950/40 border border-slate-800/50 rounded-[40px] p-12 backdrop-blur-3xl glow">
        <div className="mb-8">
          <h3 className="text-3xl font-black text-white mb-2 font-display">Trip Optimizer</h3>
        </div>

        {/* --- Source & Destination Inputs Moved Here --- */}
        <div className="mb-8 flex gap-4 relative z-50">
          <div className="flex-1 relative flex items-center">
            <input
              className="w-full bg-slate-900 border border-slate-700 p-4 pr-10 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400 transition-colors shadow-inner"
              placeholder="Source (e.g. Mumbai, India)"
              value={tripPlan.source}
              onChange={e => setTripPlan({ ...tripPlan, source: e.target.value })}
            />
            <button
              className="absolute right-3 p-2 bg-slate-800 hover:bg-slate-700 rounded-xl text-cyan-400 transition-colors flex items-center justify-center cursor-pointer"
              onClick={async () => {
                const fallback = async () => {
                  try {
                    const res = await axios.get('https://get.geojs.io/v1/ip/geo.json');
                    if (res.data && res.data.latitude && res.data.longitude) {
                      setTripPlan(prev => ({ ...prev, source: `${res.data.latitude}, ${res.data.longitude}` }));
                    } else {
                      alert("Could not get your location.");
                    }
                  } catch (err) {
                    alert("Could not get your location.");
                  }
                };
                if (navigator.geolocation) {
                  navigator.geolocation.getCurrentPosition(
                    (pos) => {
                      setTripPlan(prev => ({ ...prev, source: `${pos.coords.latitude}, ${pos.coords.longitude}` }));
                    },
                    (err) => fallback(),
                    { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
                  );
                } else {
                  fallback();
                }
              }}
              title="Use Current Location"
            >
              <MapPin size={18} />
            </button>
          </div>
          <input
            className="w-full bg-slate-900 border border-slate-700 p-4 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400 transition-colors flex-1 shadow-inner"
            placeholder="Destination (e.g. Pune, India)"
            value={tripPlan.destination}
            onChange={e => setTripPlan({ ...tripPlan, destination: e.target.value })}
          />
          <button
            className="bg-cyan-500 text-slate-950 px-8 py-3 rounded-2xl font-black uppercase tracking-widest hover:bg-cyan-400 transition-colors shadow-lg shadow-cyan-500/20"
            onClick={async () => {
              try {
                if (tripPlan.source && tripPlan.destination) {
                  const res = await axios.post(`${API_BASE}/trip/plan`, {
                    source: tripPlan.source,
                    destination: tripPlan.destination,
                    optimize_for_wait: optimizeForWait,
                    battery_pct: batteryLevel,
                    vehicle_range_km: vehicleRange
                  });
                  setTripPlan(prev => ({
                    ...prev,
                    route: res.data.route,
                    suggested: res.data.suggested_stations,
                    reachableKm: res.data.reachable_km
                  }));
                  setActiveTab('map');
                } else {
                  alert("Please enter Source and Destination");
                }
              } catch (err) {
                console.error("Trip planning error:", err);
                alert("Failed to plan trip. Check backend connection.");
              }
            }}
          >
            Plan Route
          </button>
          <button
            onClick={() => setOptimizeForWait(v => !v)}
            className={`px-6 py-3 rounded-2xl font-black uppercase tracking-widest text-sm border transition-all ${optimizeForWait
              ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400 shadow-lg shadow-emerald-500/10'
              : 'bg-slate-900/40 border-slate-700 text-slate-400 hover:text-white'
              }`}
            title="When ON, re-ranks stops by lowest predicted wait time instead of distance"
          >
            {optimizeForWait ? '⚡ Min Wait ON' : '⚡ Min Wait OFF'}
          </button>
        </div>

        <div className="mb-8">
          <p className="text-slate-400 font-medium text-lg border-b border-slate-800/50 pb-4">Configure your vehicle parameters for peak efficiency.</p>
        </div>

        <div className="space-y-10">
          <div className="space-y-6">
            <div className="flex justify-between items-end">
              <span className="text-sm font-bold text-slate-400 uppercase tracking-widest">Current Battery Level</span>
              <span className={`text-4xl font-black ${batteryLevel < 20 ? 'text-rose-500' : 'text-cyan-400'}`}>
                {batteryLevel}%
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              value={batteryLevel}
              onChange={(e) => {
                setBatteryLevel(parseInt(e.target.value));
                onRefresh();
              }}
              className="w-full h-3 bg-slate-800 rounded-full appearance-none cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-[10px] font-black text-slate-600 uppercase tracking-widest">
              <span>Critical</span>
              <span>Optimal Range</span>
              <span>Full Charge</span>
            </div>
          </div>

          {/* Vehicle Selector for Range-Aware Planning */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap size={16} className="text-cyan-400" />
              <span className="text-sm font-bold text-slate-400 uppercase tracking-widest">Select Your Vehicle</span>
              {selectedEV && vehicleStats?.range && (
                <span className="ml-auto text-xs font-black text-emerald-400 bg-emerald-900/30 px-3 py-1 rounded-full border border-emerald-500/30">
                  {vehicleStats.range} km max range
                </span>
              )}
            </div>
            <select
              className="w-full bg-slate-900 border border-slate-700 text-white rounded-2xl px-5 py-4 outline-none focus:border-cyan-400 transition-colors text-sm"
              value={selectedEV}
              onChange={(e) => {
                const val = e.target.value;
                setSelectedEV(val);
                if (!val) {
                  setVehicleStats(null);
                  setVehicleRange(400); // reset to default
                  return;
                }
                const [brand, ...rest] = val.split(' - ');
                const model = rest.join(' - ');
                if (vehicleData[brand]) {
                  const stats = vehicleData[brand].find(m => m.model === model);
                  setVehicleStats(stats || null);
                  if (stats?.range && !isNaN(parseFloat(stats.range))) {
                    setVehicleRange(parseFloat(stats.range));
                  }
                }
              }}
            >
              <option value="">— Choose a vehicle to calculate reachability —</option>
              {Object.keys(vehicleData).map(b => (
                <optgroup key={b} label={b}>
                  {vehicleData[b].map(m => (
                    <option key={`${b} - ${m.model}`} value={`${b} - ${m.model}`}>
                      {b} {m.model}{m.range ? ` · ${m.range} km` : ''}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>

            {/* Live vehicle stats cards */}
            {vehicleStats ? (
              <div className="grid grid-cols-3 gap-4">
                <div className="p-5 bg-slate-900/30 rounded-3xl border border-cyan-500/20 relative overflow-hidden">
                  <div className="absolute right-4 top-4 w-8 h-8 bg-cyan-400/5 rounded-full blur-xl" />
                  <Battery className="text-cyan-400 mb-3" size={18} />
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Max Range</p>
                  <p className="text-2xl font-black text-white">{vehicleStats.range ?? '—'}<span className="text-sm text-slate-400 ml-1">km</span></p>
                  <p className="text-[10px] text-slate-600 mt-1">at 100% battery</p>
                </div>
                <div className="p-5 bg-slate-900/30 rounded-3xl border border-violet-500/20 relative overflow-hidden">
                  <div className="absolute right-4 top-4 w-8 h-8 bg-violet-400/5 rounded-full blur-xl" />
                  <Zap className="text-violet-400 mb-3" size={18} />
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Fast Charge</p>
                  <p className="text-2xl font-black text-white">{vehicleStats.fastCharge ?? '—'}<span className="text-sm text-slate-400 ml-1">km/h</span></p>
                  <p className="text-[10px] text-slate-600 mt-1">charge speed</p>
                </div>
                <div className="p-5 bg-slate-900/30 rounded-3xl border border-emerald-500/20 relative overflow-hidden">
                  <div className="absolute right-4 top-4 w-8 h-8 bg-emerald-400/5 rounded-full blur-xl" />
                  <TrendingUp className="text-emerald-400 mb-3" size={18} />
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Efficiency</p>
                  <p className="text-2xl font-black text-white">{vehicleStats.efficiency ?? '—'}<span className="text-sm text-slate-400 ml-1">Wh/km</span></p>
                  <p className="text-[10px] text-slate-600 mt-1">consumption</p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                <div className="p-5 bg-slate-900/30 rounded-3xl border border-slate-800/50">
                  <Zap className="text-amber-400 mb-3" size={18} />
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Target SoC</p>
                  <p className="text-xl font-bold text-white">80%</p>
                </div>
                <div className="p-5 bg-slate-900/30 rounded-3xl border border-slate-800/50">
                  <Clock className="text-violet-400 mb-3" size={18} />
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Departure</p>
                  <p className="text-xl font-bold text-white">ASAP</p>
                </div>
                <div className="p-5 bg-slate-900/30 rounded-3xl border border-slate-800/50">
                  <TrendingUp className="text-emerald-400 mb-3" size={18} />
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Default Range</p>
                  <p className="text-xl font-bold text-white">400<span className="text-sm text-slate-400 ml-1">km</span></p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <h4 className="text-xl font-bold text-white px-4 font-display">Recommended Charging Stations</h4>
        <div className="grid grid-cols-1 gap-4">
          {recommendations.map((rec, idx) => {
            const congColor = rec.congestionLevel === 'Busy' ? 'text-rose-400' : rec.congestionLevel === 'Moderate' ? 'text-amber-400' : 'text-emerald-400';
            const congBg = rec.congestionLevel === 'Busy' ? 'bg-rose-500/10 border-rose-500/30' : rec.congestionLevel === 'Moderate' ? 'bg-amber-500/10 border-amber-500/30' : 'bg-emerald-500/10 border-emerald-500/30';
            return (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                key={`${rec.stationID}-${idx}`}
                className="flex items-center justify-between p-6 bg-slate-900/20 border border-slate-800/40 rounded-[32px] hover:bg-slate-900/40 transition-all group cursor-pointer relative overflow-hidden"
              >
                <div className="absolute left-0 top-0 w-1.5 h-full bg-transparent group-hover:bg-cyan-400 transition-all" />
                <div className="flex items-center gap-5 relative z-10 flex-1 min-w-0">
                  <div className="w-12 h-12 bg-slate-950 rounded-2xl flex items-center justify-center text-cyan-400 border border-slate-800 group-hover:border-cyan-400/50 transition-all shadow-lg flex-shrink-0">
                    <Zap size={22} />
                  </div>
                  <div className="min-w-0">
                    <h5 className="text-base font-black text-white mb-0.5 truncate">{rec.stationID}</h5>
                    <p className="text-xs text-slate-400 mb-2">{[rec.city, rec.state].filter(Boolean).join(', ') || 'India'}{rec.operator ? ` · ${rec.operator}` : ''}</p>
                    <div className="flex flex-wrap items-center gap-2">
                      {rec.chargerType && (
                        <span className="text-[10px] font-bold bg-slate-800 text-slate-300 px-2 py-0.5 rounded-md uppercase tracking-wide">{rec.chargerType}</span>
                      )}
                      {rec.powerKw > 0 && (
                        <span className="text-[10px] font-bold bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded-md">{rec.powerKw} kW</span>
                      )}
                      <span className={`text-[10px] font-bold border px-2 py-0.5 rounded-md ${congBg} ${congColor}`}>
                        {rec.congestionLevel || 'Low'}
                      </span>
                      <span className="text-[10px] font-bold text-slate-500">
                        {rec.availableChargers}/{rec.totalChargers} free
                      </span>
                      {rec.distanceKm != null && (
                        <span className="text-[10px] font-bold text-slate-500 flex items-center gap-1">
                          <Navigation size={9} /> {rec.distanceKm} km
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="text-right relative z-10 flex-shrink-0 ml-4">
                  <div className="mb-1">
                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest block mb-1">Score</span>
                    <p className="text-3xl font-black text-cyan-400">{Math.max(0, Math.round(rec.score))}</p>
                  </div>
                  {rec.predictedWait > 0 && (
                    <p className="text-xs text-rose-400 font-bold mb-2">~{rec.predictedWait}m wait</p>
                  )}
                  <button className="px-5 py-2 bg-slate-800 hover:bg-white hover:text-slate-950 text-white rounded-xl text-xs font-black uppercase tracking-widest transition-all shadow-xl">
                    Route Now
                  </button>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Battery-Aware Suggested Stations from Trip Plan */}
        {tripPlan.suggested && tripPlan.suggested.length > 0 && (() => {
          const reachable = tripPlan.suggested.filter(s => s.reachable !== false);
          const outOfRange = tripPlan.suggested.filter(s => s.reachable === false);
          const displayKm = tripPlan.reachableKm ?? reachableKm;
          const maxDist = Math.max(...tripPlan.suggested.map(s => s.km_to_station ?? s.dist_to_start ?? 0), 1);
          const barFill = Math.min(100, (displayKm / maxDist) * 100);

          const StationCard = ({ st, idx, dimmed }) => {
            const level = st.congestion_level || 'Low';
            const wait = st.predicted_wait_minutes ?? 0;
            const pct = st.congestion_pct ?? 0;
            const dist = st.km_to_station ?? st.dist_to_start ?? 0;
            const levelColors = {
              Low: { badge: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30', bar: 'bg-emerald-500', border: 'border-emerald-500/20' },
              Moderate: { badge: 'bg-amber-500/15 text-amber-400 border-amber-500/30', bar: 'bg-amber-400', border: 'border-amber-500/20' },
              Busy: { badge: 'bg-rose-500/15 text-rose-400 border-rose-500/30', bar: 'bg-rose-500', border: 'border-rose-500/20' },
            };
            const colors = levelColors[level] || levelColors.Low;
            return (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: dimmed ? 0.45 : 1, y: 0 }}
                transition={{ delay: idx * 0.04 }}
                key={`sug-${idx}`}
                className={`p-5 bg-slate-900/20 border rounded-3xl hover:bg-slate-900/50 transition-all cursor-pointer relative overflow-hidden ${dimmed ? 'border-slate-700/40' : colors.border}`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-center gap-4 flex-1 min-w-0">
                    <div className={`w-11 h-11 rounded-2xl flex items-center justify-center border flex-shrink-0 ${dimmed ? 'bg-slate-800/60 text-slate-500 border-slate-700' : colors.badge}`}>
                      <Zap size={18} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-white font-bold truncate text-sm">{st.stationID}</p>
                      <p className="text-slate-500 text-xs mt-0.5">
                        {st.city} &nbsp;·&nbsp;
                        <span className={dimmed ? 'text-rose-400 font-bold' : 'text-cyan-400 font-semibold'}>{dist.toFixed(1)} km away</span>
                      </p>
                      <div className="mt-2 w-full max-w-[160px]">
                        <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                          <div className={`h-full ${dimmed ? 'bg-slate-600' : colors.bar} transition-all`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1.5 flex-shrink-0">
                    {/* Primary badge — context-aware */}
                    {optimizeForWait ? (
                      <span className={`text-[10px] font-black uppercase tracking-widest px-2.5 py-1 rounded-full border ${wait === 0 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : wait < 10 ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                            : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                        }`}>
                        ⚡ {wait > 0 ? `~${wait}m wait` : 'No wait'}
                      </span>
                    ) : dimmed ? (
                      <span className="text-[10px] font-black uppercase tracking-widest px-2.5 py-1 rounded-full border bg-rose-500/10 text-rose-400 border-rose-500/30">
                        Out of Range
                      </span>
                    ) : (
                      <span className="text-[10px] font-black uppercase tracking-widest px-2.5 py-1 rounded-full border bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
                        ✓ Reachable
                      </span>
                    )}
                    <span className={`text-[10px] font-bold ${dimmed ? 'text-slate-600' : colors.badge.split(' ')[1]}`}>
                      {level}
                    </span>
                    {/* Secondary info line — swap when in wait mode */}
                    {optimizeForWait ? (
                      <span className={`text-[10px] font-bold ${dimmed ? 'text-slate-600' : 'text-slate-400'}`}>
                        {dimmed ? '⚠ Out of range' : '✓ Reachable'}
                      </span>
                    ) : (
                      <span className="text-[10px] text-slate-500 font-bold">
                        {wait > 0 ? `~${wait}m wait` : 'No wait'}
                      </span>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          };



          return (
            <div className="mt-8 space-y-5">
              {/* Header + Sort badge */}
              <div className="flex items-center justify-between px-2">
                <h4 className="text-xl font-bold text-white font-display">Stops Along Route</h4>
                <span className={`text-xs font-bold uppercase tracking-widest px-3 py-1 rounded-full border ${optimizeForWait
                  ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30'
                  : 'text-cyan-400 bg-cyan-400/10 border-cyan-400/30'
                  }`}>
                  {optimizeForWait ? '⚡ Min Wait' : '📍 By Distance'}
                </span>
              </div>

              {/* Battery Range Bar */}
              <div className="p-5 bg-slate-900/40 rounded-3xl border border-slate-800/60 space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <Battery size={16} className={batteryLevel < 20 ? 'text-rose-400' : 'text-cyan-400'} />
                    <span className="text-sm font-bold text-white">Battery Range</span>
                    {selectedEV ? (
                      <span className="text-xs text-slate-400 font-semibold truncate max-w-[180px]" title={selectedEV}>
                        · {selectedEV}
                      </span>
                    ) : (
                      <span className="text-xs text-amber-500/70 italic">— pick a vehicle above for accurate range</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-600 font-mono">
                      {batteryLevel}% × {vehicleRange} km
                    </span>
                    <span className="text-slate-700">=</span>
                    <span className={`text-sm font-black ${batteryLevel < 20 ? 'text-rose-400' : 'text-emerald-400'}`}>
                      ~{displayKm} km
                    </span>
                  </div>
                </div>
                <div className="relative w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${barFill}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    className={`h-full rounded-full ${batteryLevel < 20 ? 'bg-gradient-to-r from-rose-600 to-rose-400' :
                      batteryLevel < 50 ? 'bg-gradient-to-r from-amber-500 to-yellow-400' :
                        'bg-gradient-to-r from-cyan-500 to-emerald-400'
                      } shadow-[0_0_8px_rgba(34,211,238,0.4)]`}
                  />
                  {/* station tick marks */}
                  {tripPlan.suggested.map((s, i) => {
                    const d = s.km_to_station ?? s.dist_to_start ?? 0;
                    const x = Math.min(100, (d / maxDist) * 100);
                    return (
                      <div
                        key={i}
                        className={`absolute top-0 bottom-0 w-0.5 ${s.reachable !== false ? 'bg-white/50' : 'bg-rose-500/70'
                          }`}
                        style={{ left: `${x}%` }}
                        title={`${s.stationID} (${d.toFixed(1)} km)`}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-[10px] font-bold text-slate-600 uppercase tracking-wider">
                  <span>Start</span>
                  <span className={batteryLevel < 20 ? 'text-rose-400' : 'text-cyan-400'}>{displayKm} km</span>
                  <span>Destination</span>
                </div>
              </div>

              {/* ✅ Reachable Stations */}
              {reachable.length > 0 && (
                <div className="space-y-3">
                  <p className="text-xs font-black uppercase tracking-widest text-emerald-400 px-2 flex items-center gap-2">
                    <CheckCircle2 size={13} /> {reachable.length} Reachable Station{reachable.length > 1 ? 's' : ''}
                  </p>
                  {reachable.slice(0, 8).map((st, idx) => (
                    <StationCard key={`r-${idx}`} st={st} idx={idx} dimmed={false} />
                  ))}
                </div>
              )}

              {/* ⚠️ Out of Range Stations */}
              {outOfRange.length > 0 && (
                <div className="space-y-3">
                  <p className="text-xs font-black uppercase tracking-widest text-rose-400 px-2 flex items-center gap-2">
                    <AlertCircle size={13} /> {outOfRange.length} Out of Range — charge first
                  </p>
                  {outOfRange.slice(0, 6).map((st, idx) => (
                    <StationCard key={`o-${idx}`} st={st} idx={idx} dimmed={true} />
                  ))}
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </motion.div>
  );
}

function NavItem({ icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-4 px-6 py-4 rounded-2xl transition-all relative group ${active
        ? 'bg-cyan-400/10 text-cyan-400'
        : 'text-slate-500 hover:text-slate-300 hover:bg-slate-900/40'
        }`}
    >
      {active && <motion.div layoutId="nav-glow" className="absolute inset-0 bg-cyan-400/5 blur-xl rounded-2xl" />}
      <span className={`${active ? 'text-cyan-400' : 'text-slate-600 group-hover:text-slate-400'} transition-colors`}>{icon}</span>
      <span className="font-bold tracking-tight">{label}</span>
      {active && <div className="absolute left-0 w-1 h-6 bg-cyan-400 rounded-r-full shadow-[0_0_10px_rgba(34,211,238,1)]" />}
    </button>
  );
}

function StatCard({ title, value, sub, icon, color }) {
  const colors = {
    cyan: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    violet: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
  };

  return (
    <div className="bg-slate-950/40 border border-slate-800/50 rounded-[32px] p-6 backdrop-blur-xl group hover:border-slate-700/50 transition-all cursor-default">
      <div className="flex justify-between items-start mb-6">
        <div className={`p-3 rounded-2xl ${colors[color]} border transition-transform group-hover:scale-110`}>
          {icon}
        </div>
      </div>
      <div>
        <p className="text-3xl font-black text-white mb-1 group-hover:translate-x-1 transition-transform">{value}</p>
        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{title}</p>
        <div className="mt-4 pt-4 border-t border-slate-800/50">
          <p className="text-[10px] font-medium text-slate-600 italic">"{sub}"</p>
        </div>
      </div>
    </div>
  );
}

function LoadItem({ label, percent, color }) {
  const colors = {
    cyan: 'bg-cyan-400',
    emerald: 'bg-emerald-400',
    amber: 'bg-amber-400'
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest">
        <span className="text-slate-500">{label}</span>
        <span className="text-slate-300">{percent}%</span>
      </div>
      <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          className={`h-full ${colors[color]}`}
        />
      </div>
    </div>
  );
}

function SmartRouteList({ routes, loading, selectedIdx, onSelect }) {
  const rankEmoji = ['🥇', '🥈', '🥉'];
  const levelColors = {
    Low: { badge: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30', bar: 'bg-emerald-500', border: 'border-emerald-500/20', dot: 'bg-emerald-400' },
    Moderate: { badge: 'bg-amber-500/15 text-amber-400 border-amber-500/30', bar: 'bg-amber-400', border: 'border-amber-500/20', dot: 'bg-amber-400' },
    Busy: { badge: 'bg-rose-500/15 text-rose-400 border-rose-500/30', bar: 'bg-rose-500', border: 'border-rose-500/20', dot: 'bg-rose-400' },
  };

  return (
    <div className="w-full bg-slate-900/30 border border-slate-800/50 rounded-[30px] p-6 backdrop-blur-3xl">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-xl font-bold text-white font-display">Optimal Charging Routes</h3>
          <p className="text-xs text-slate-500 mt-1">Ranked by drive time + live congestion · Updates every 30s</p>
        </div>
        {loading && (
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-bold">
            <div className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
            Fetching routes...
          </div>
        )}
        {!loading && routes.length > 0 && (
          <span className="text-xs text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 px-3 py-1 rounded-full font-bold">
            {routes.length} stations found
          </span>
        )}
      </div>

      {!loading && routes.length === 0 && (
        <p className="text-slate-500 text-sm text-center py-8">No EV stations found within 10 km. Allow GPS or drag the map pin.</p>
      )}

      <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1" style={{ scrollbarWidth: 'thin', scrollbarColor: '#334155 transparent' }}>
        {routes.map((route, idx) => {
          const level = route.congestion_level || 'Low';
          const colors = levelColors[level] || levelColors.Low;
          const isSelected = selectedIdx === idx;
          const score = route.score ?? 0;
          const maxScore = routes[0]?.score ?? 100;

          return (
            <button
              key={idx}
              onClick={() => onSelect(idx, route)}
              className={`w-full text-left p-5 rounded-2xl border transition-all duration-200 relative overflow-hidden group ${isSelected
                ? 'bg-slate-800/80 border-cyan-400/40 shadow-lg shadow-cyan-400/10'
                : `bg-slate-900/20 ${colors.border} hover:bg-slate-800/40 hover:border-slate-700`
                }`}
            >
              {/* Selected glow strip */}
              {isSelected && <div className="absolute left-0 top-0 h-full w-1 bg-cyan-400 rounded-l-2xl" />}

              <div className="flex items-start gap-4 pl-1">
                {/* Rank Badge */}
                <div className="flex-shrink-0 text-center">
                  <span className="text-2xl">{rankEmoji[idx] || `#${idx + 1}`}</span>
                  <div className="text-[9px] text-slate-500 font-bold mt-0.5">Score</div>
                  <div className="text-sm font-black text-cyan-400">{Math.round(score)}</div>
                </div>

                {/* Station Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="text-white font-bold text-base truncate leading-snug">{route.name}</p>
                      <p className="text-slate-500 text-xs">{route.city}{route.state ? `, ${route.state}` : ''}</p>
                    </div>
                    <span className={`flex-shrink-0 text-[10px] font-black uppercase tracking-widest px-2 py-1 rounded-full border ${colors.badge}`}>
                      {level}
                    </span>
                  </div>

                  {/* Metrics row */}
                  <div className="flex items-center gap-4 mt-2 flex-wrap">
                    <span className="flex items-center gap-1 text-xs text-slate-400 font-medium">
                      🚗 <span className="text-white font-bold">{Math.round(route.drive_minutes)} min</span>
                      <span className="text-slate-600">·</span>
                      <span>{route.distance_km} km</span>
                    </span>
                    <span className="flex items-center gap-1 text-xs text-slate-400 font-medium">
                      🔋 <span className={route.available_chargers > 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                        {route.available_chargers}/{route.total_chargers}
                      </span>
                      <span>free</span>
                    </span>
                    {route.predicted_wait_minutes > 0 && (
                      <span className="flex items-center gap-1 text-xs text-rose-400 font-medium">
                        ⏱ ~{route.predicted_wait_minutes} min wait
                      </span>
                    )}
                  </div>

                  {/* Score bar */}
                  <div className="mt-2.5 w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${colors.bar} transition-all duration-700`}
                      style={{ width: `${Math.max(0, Math.min(100, (score / Math.max(maxScore, 1)) * 100))}%` }}
                    />
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function NearbyStations({ lat, lng }) {
  const [nearby, setNearby] = useState([]);
  const [loading, setLoading] = useState(true);
  const [congestionMap, setCongestionMap] = useState({});

  useEffect(() => {
    const fetchNearby = async () => {
      try {
        setLoading(true);
        const res = await axios.get(`${API_BASE}/stations/nearby?lat=${lat}&lng=${lng}&radius=10`);
        if (res.data.stations) {
          const stationList = res.data.stations;
          setNearby(stationList);

          // Fetch congestion per ACN station for enrichment.
          // Nearby stations use Indian EV CSV names, not ACN IDs.
          // We'll use available charger count as a proxy but also call /stations/congestion
          // for any matching ACN station IDs we know.
          try {
            const congRes = await axios.get(`${API_BASE}/stations/congestion`);
            if (congRes.data?.congestion) {
              const map = {};
              congRes.data.congestion.forEach(s => { map[s.station_id] = s; });
              setCongestionMap(map);
            }
          } catch (_) { }
        }
      } catch (err) {
        console.error("Failed to fetch nearby stations", err);
      } finally {
        setLoading(false);
      }
    };
    if (lat && lng) fetchNearby();
  }, [lat, lng]);

  if (loading) {
    return (
      <div className="w-full h-40 bg-slate-900/40 border border-slate-800/50 backdrop-blur-3xl rounded-[30px] flex items-center justify-center">
        <Loader className="text-cyan-400 animate-spin" size={24} />
      </div>
    );
  }

  if (nearby.length === 0) {
    return (
      <div className="w-full p-6 bg-slate-900/40 border border-slate-800/50 backdrop-blur-3xl rounded-[30px] text-center text-slate-400">
        No EV charging stations found within 10km of your location.
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="flex items-center justify-between px-4 mb-4">
        <h3 className="text-xl font-bold text-white font-display">Nearby Stations</h3>
        <span className="text-sm font-medium text-cyan-400 bg-cyan-400/10 px-3 py-1 rounded-full border border-cyan-400/20 shadow-[0_0_10px_rgba(34,211,238,0.2)]">
          {nearby.length} Found
        </span>
      </div>

      <div className="flex gap-4 overflow-x-auto pb-6 snap-x snap-mandatory hide-scrollbar pl-4 pr-12 pb-4 pt-2 -mx-4" style={{ WebkitOverflowScrolling: 'touch', scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
        {nearby.map((station, idx) => {
          // Determine congestion: use real-time data if available, else derive from availability
          const totalChargers = station.total_chargers || 1;
          const available = station.available ?? totalChargers;
          const occupancyPct = Math.round(((totalChargers - available) / totalChargers) * 100);
          const congLevel = occupancyPct < 40 ? 'Low' : occupancyPct < 70 ? 'Moderate' : 'Busy';
          const barColor = congLevel === 'Low' ? 'bg-emerald-500' : congLevel === 'Moderate' ? 'bg-amber-400' : 'bg-rose-500';
          const badgeColor = congLevel === 'Low'
            ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
            : congLevel === 'Moderate'
              ? 'bg-amber-500/15 text-amber-400 border-amber-500/30'
              : 'bg-rose-500/15 text-rose-400 border-rose-500/30';
          const topBarColor = station.available > 0 ? 'bg-emerald-500' : 'bg-rose-500';

          return (
            <div
              key={station.id || idx}
              className="snap-start shrink-0 w-72 bg-gradient-to-br from-slate-900/60 to-slate-950/60 border border-slate-800/80 rounded-3xl p-5 shadow-xl backdrop-blur-xl relative overflow-hidden group hover:border-cyan-500/30 transition-all duration-300"
            >
              {/* Status top bar */}
              <div className={`absolute top-0 left-0 w-full h-1 ${topBarColor} opacity-80`} />

              <div className="flex justify-between items-start mb-3">
                <h4 className="text-white font-bold text-lg truncate pr-2 w-44" title={station.name}>
                  {station.name}
                </h4>
                <div className="bg-slate-800/80 rounded-lg px-2 py-1 text-xs font-bold text-cyan-400 whitespace-nowrap border border-slate-700">
                  {station.distance} km
                </div>
              </div>

              {/* Congestion badge */}
              <div className="mb-3">
                <span className={`text-[10px] font-black uppercase tracking-widest px-2.5 py-1 rounded-full border ${badgeColor}`}>
                  ⚡ {congLevel} Congestion
                </span>
                <div className="mt-2 w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                  <div className={`h-full ${barColor} transition-all`} style={{ width: `${occupancyPct}%` }} />
                </div>
              </div>

              <div className="space-y-2 mb-4">
                <div className="flex items-center gap-2 text-slate-300 text-sm">
                  <Zap size={14} className="text-amber-400" />
                  <span className="truncate">{station.charger_type}</span>
                </div>
                <div className="flex items-center gap-2 text-slate-300 text-sm">
                  <Battery size={14} className="text-blue-400" />
                  <span>{station.available} / {station.total_chargers} Available</span>
                </div>
                {station.wait_time_mins > 0 && (
                  <div className="flex items-center gap-2 text-rose-400 text-sm font-medium">
                    <Clock size={14} />
                    <span>~{station.wait_time_mins} min wait</span>
                  </div>
                )}
              </div>

              <button className={`w-full py-2.5 rounded-xl text-sm font-bold transition-all shadow-md ${station.available > 0
                ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 hover:bg-cyan-500 hover:text-slate-950"
                : "bg-slate-800/50 text-slate-500 cursor-not-allowed border border-slate-800"
                }`}>
                {station.available > 0 ? "Navigate Here" : "Occupied"}
              </button>
            </div>
          );
        })}
      </div>

      <style dangerouslySetInnerHTML={{
        __html: `
        .hide-scrollbar::-webkit-scrollbar {
          display: none;
        }
      `}} />
    </div>
  );
}

export default App;


