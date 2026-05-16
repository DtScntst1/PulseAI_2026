import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import pickle
import os

def generate_sensor_data(n_samples=5000):
    np.random.seed(42)
    
    machine_ids = [f"MAC-{i:03d}" for i in range(1, 51)]  # 50 machines
    
    data = []
    for m_id in machine_ids:
        # Each machine has a random operating baseline
        base_temp = np.random.uniform(60, 80)
        base_vib = np.random.uniform(2.0, 5.0)
        base_press = np.random.uniform(100, 120)
        base_rpm = np.random.uniform(1400, 1600)
        
        # Determine if machine is healthy or degrading
        is_degrading = np.random.choice([True, False], p=[0.3, 0.7])
        
        # Time steps for each machine (between 50 to 150 days of records)
        n_days = np.random.randint(50, 150)
        
        rul = np.linspace(n_days, 0, n_days) if is_degrading else np.random.randint(100, 300, n_days)
        
        for day in range(n_days):
            degradation_factor = (n_days - day) / n_days if is_degrading else 1.0
            
            temp = base_temp + np.random.normal(0, 2) + (20 * (1 - degradation_factor) if is_degrading else 0)
            vib = base_vib + np.random.normal(0, 0.5) + (5 * (1 - degradation_factor) if is_degrading else 0)
            press = base_press + np.random.normal(0, 5) + (30 * (1 - degradation_factor) if is_degrading else 0)
            rpm = base_rpm + np.random.normal(0, 20) - (200 * (1 - degradation_factor) if is_degrading else 0)
            
            # Label as anomaly if values are extreme
            is_anomaly = 1 if (temp > 95 or vib > 8.0 or press > 145 or rpm < 1200) else 0
            
            data.append({
                'MachineID': m_id,
                'Day': day,
                'Temperature': temp,
                'Vibration': vib,
                'Pressure': press,
                'RPM': rpm,
                'Is_Anomaly': is_anomaly,
                'RUL': rul[day]
            })
            
    df = pd.DataFrame(data)
    return df

def train_and_save_model():
    print("Generating synthetic IoT sensor data...")
    df = generate_sensor_data()
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sensor_data.csv', index=False)
    
    features = ['Temperature', 'Vibration', 'Pressure', 'RPM']
    X = df[features]
    y_rul = df['RUL']
    
    print("Training Isolation Forest for Anomaly Detection...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)
    
    print("Training XGBoost Regressor for RUL Prediction...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y_rul)
    
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/anomaly_model.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
        
    with open('models/rul_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
        
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    print("Done! Data and models saved.")

if __name__ == "__main__":
    train_and_save_model()
