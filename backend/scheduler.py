import pandas as pd
from datetime import timedelta

class ChargingScheduler:
    def __init__(self, data_frame):
        self.df = data_frame

    def predict_wait_time(self, station_id, arrival_time):
        """
        Simulates wait time by looking at sessions that will still be 
        occupying the station at arrival_time.
        """
        station_sessions = self.df[self.df['stationID'] == station_id]
        # Active sessions at arrival time
        active = station_sessions[(station_sessions['connectionTime'] <= arrival_time) & 
                                  (station_sessions['disconnectTime'] > arrival_time)]
        
        if active.empty:
            return 0
        
        # Simplistic queue model: wait until the soonest session ends
        wait_until = active['disconnectTime'].min()
        wait_seconds = (wait_until - arrival_time).total_seconds()
        return max(0, wait_seconds / 60) # in minutes

    def suggest_optimal_stop(self, current_location, destination, battery_level, stations_data):
        """
        stations_data: list of dicts {id, lat, lon, distance_from_current}
        Optimizes for total trip time: Travel Time + Wait Time + Charging Time.
        """
        recommendations = []
        
        for station in stations_data:
            # Mocking travel time (1 min per km)
            travel_time = station['distance'] 
            # Mocking wait time prediction using historical average if current is not available
            predicted_wait = 15 # default 
            
            total_delay = travel_time + predicted_wait
            
            recommendations.append({
                'stationID': station['id'],
                'predictedWait': predicted_wait,
                'travelTime': travel_time,
                'totalDelay': total_delay,
                'score': 100 - total_delay # Higher is better
            })
        
        # Sort by total delay
        return sorted(recommendations, key=lambda x: x['totalDelay'])

if __name__ == "__main__":
    from data_processor import load_and_preprocess
    DATA_PATH = r"c:\Users\naizan\Desktop\projects\group11\acn-data (1) (1).csv"
    df = load_and_preprocess(DATA_PATH)
    scheduler = ChargingScheduler(df)
    
    # Test prediction
    test_time = df['connectionTime'].iloc[0] + timedelta(hours=1)
    stat_id = df['stationID'].iloc[0]
    wait = scheduler.predict_wait_time(stat_id, test_time)
    print(f"Predicted wait for {stat_id} at {test_time}: {wait:.2f} mins")
