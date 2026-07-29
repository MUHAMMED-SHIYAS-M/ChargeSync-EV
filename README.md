# ⚡ ChargeSync: AI-Powered EV Charging Optimization Platform

ChargeSync is an intelligent EV charging management system that uses **Artificial Intelligence, Machine Learning, and Data Analytics** to optimize electric vehicle charging operations. The system predicts charging demand, forecasts energy consumption, detects congestion, and provides efficient charging slot management.

The platform helps reduce charging delays, balance station loads, and improve user experience through AI-driven predictions and real-time monitoring.

---

## 🚀 Key Features

### 🔋 1. EV Charging Demand Forecasting
- Predicts future charging energy demand using time-series models.
- Uses historical charging session data to identify usage patterns.
- Implemented using:
  - LSTM Neural Network
  - ARIMA Statistical Model

### 🚦 2. Charging Station Congestion Prediction
- Analyzes charging station occupancy patterns.
- Predicts possible congestion periods.
- Helps users select less crowded charging stations.

### 📊 3. Real-Time Monitoring Dashboard
- Displays charging station status.
- Shows available and occupied charging slots.
- Provides analytics for administrators.

### 🧠 4. AI-Based Optimization
- Uses machine learning models to improve charging efficiency.
- Supports better scheduling and resource utilization.

### 📈 5. Data Analytics
- Extracts useful insights from EV charging data.
- Performs feature engineering and visualization.

---

# 🏗️ System Architecture

```
                 User
                  |
                  |
          React Web Interface
                  |
                  |
             FastAPI Backend
                  |
        -----------------------
        |                     |
  ML Prediction Engine     Database
        |                     |
 LSTM / ARIMA Models      SQLite
        |
 Charging Demand &
 Congestion Prediction
```

---

# 🛠️ Technologies Used

## Frontend
- React.js
- Vite
- Tailwind CSS
- Axios
- React Leaflet (Map Visualization)

## Backend
- FastAPI
- Python
- Uvicorn
- SQLAlchemy

## Machine Learning
- TensorFlow / Keras
- Scikit-learn
- Statsmodels
- Pandas
- NumPy

## Database
- SQLite (Development)
- PostgreSQL (Future Deployment)

---

# 📂 Project Structure

```
ChargeSync/
│
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── models/
│   │   ├── lstm_service.py
│   │   └── congestion_model.py
│   ├── data_processor.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   └── package.json
│
├── dataset/
│   └── ACN_Data.csv
│
├── README.md
└── requirements.txt
```

---

# 📊 Dataset

The project uses the **Adaptive Charging Network (ACN) dataset** containing EV charging session information.

### Dataset Features:
- Station ID
- Charger ID
- Connector Type
- Start Time
- End Time
- Charging Duration
- Energy Delivered (kWh)
- Charging Power
- Session Information

### Data Processing:
- Missing value handling
- Timestamp conversion
- Feature extraction:
  - Hour
  - Day
  - Weekday
  - Month
- Lag feature generation
- Rolling statistics calculation

---

# 🤖 Machine Learning Models

## 1. LSTM Demand Forecasting

**Purpose:**
Predict future energy demand from historical charging patterns.

Architecture:

```
Input Sequence
      |
LSTM (64 units)
      |
Dropout (0.2)
      |
LSTM (32 units)
      |
Dropout (0.2)
      |
Dense Layer
      |
Predicted Energy Demand
```

Parameters:
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Sequence Length: 10

---

## 2. ARIMA Forecasting Model

Used as a statistical baseline model.

Model:

```
ARIMA(5,1,0)
```

Evaluation Metrics:
- RMSE
- MAE
- MAPE

---

# 📌 Installation Guide

## Clone Repository

```bash
git clone https://github.com/your-username/ChargeSync.git

cd ChargeSync
```

---

# Backend Setup

Navigate to backend:

```bash
cd backend
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend:

```bash
uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

# Frontend Setup

Navigate to frontend:

```bash
cd frontend
```

Install packages:

```bash
npm install
```

Start development server:

```bash
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# 📈 Model Performance

Evaluation metrics used:

| Metric | Purpose |
|--------|---------|
| RMSE | Measures prediction error |
| MAE | Measures average error |
| R² Score | Measures model accuracy |
| MAPE | Measures percentage error |

---

# 🔮 Future Enhancements

- Integration with real-time IoT charging stations
- Cloud deployment using AWS/GCP
- Reinforcement learning based charging optimization
- Mobile application support
- Dynamic pricing prediction
- Multi-station energy balancing

---

# 👨‍💻 Contributors

**ChargeSync Development Team**

AI & Data Science Project

---

# 📄 License

This project is developed for academic and research purposes.
