# PulseAI ⚙️: Industrial Predictive Maintenance System

PulseAI is an advanced, end-to-end Machine Learning system designed for the Industrial Internet of Things (IIoT). It leverages synthetic high-frequency sensor data to detect machine anomalies in real-time and predict the Remaining Useful Life (RUL) of industrial equipment, preventing millions of dollars in unexpected downtime.

## 🌟 Key Features
- **Real-Time Anomaly Detection:** Uses `IsolationForest` to monitor streaming sensor data (Vibration, Temperature, Pressure, RPM) and flag irregular operational states.
- **RUL Forecasting:** Implements an `XGBoost` regressor to calculate exactly how many days a machine has left before failure.
- **Explainable AI (XAI):** Integrated with `SHAP` to provide a "Diagnostic Center", explaining *why* a machine is predicted to fail (e.g., "Vibration is 40% higher than baseline").
- **Live Control Room Dashboard:** A sleek, cyber-industrial `Streamlit` interface simulating a factory control room.

## 🚀 Live Demo
Experience the factory control room live:
**[Insert Streamlit Link Here]**

## 💻 Tech Stack
- **Machine Learning:** XGBoost, Scikit-Learn (Isolation Forest), Pandas, NumPy
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Frontend & Deployment:** Streamlit (Custom Dark CSS), Streamlit Cloud
- **Data:** Synthetic IoT Sensor Generator built-in

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DtScntst1/PulseAI.git
   cd PulseAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *Note: The app is self-contained. It will automatically generate synthetic sensor data and train the ML models on its first run.*

## 📓 Kaggle Notebook
View the Exploratory Data Analysis (EDA) and model training pipeline on Kaggle:
**[https://www.kaggle.com/code/mhsn21/pulseai-predictive-maintenance-iot-analytics](https://www.kaggle.com/code/mhsn21/pulseai-predictive-maintenance-iot-analytics)**
