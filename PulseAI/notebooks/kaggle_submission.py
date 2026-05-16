import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# --- 1. Synthetic IoT Data Generation ---
print("Generating IoT Sensor Data...")
np.random.seed(42)
machine_ids = [f"MAC-{i:03d}" for i in range(1, 101)]

data = []
for m_id in machine_ids:
    base_temp = np.random.uniform(60, 80)
    base_vib = np.random.uniform(2.0, 5.0)
    base_press = np.random.uniform(100, 120)
    base_rpm = np.random.uniform(1400, 1600)
    
    is_degrading = np.random.choice([True, False], p=[0.4, 0.6])
    n_days = np.random.randint(50, 150)
    rul = np.linspace(n_days, 0, n_days) if is_degrading else np.random.randint(100, 300, n_days)
    
    for day in range(n_days):
        dfac = (n_days - day) / n_days if is_degrading else 1.0
        
        temp = base_temp + np.random.normal(0, 2) + (25 * (1 - dfac) if is_degrading else 0)
        vib = base_vib + np.random.normal(0, 0.5) + (6 * (1 - dfac) if is_degrading else 0)
        press = base_press + np.random.normal(0, 5) + (35 * (1 - dfac) if is_degrading else 0)
        rpm = base_rpm + np.random.normal(0, 20) - (250 * (1 - dfac) if is_degrading else 0)
        
        data.append({
            'MachineID': m_id, 'Day': day, 'Temperature': temp,
            'Vibration': vib, 'Pressure': press, 'RPM': rpm, 'RUL': rul[day]
        })

df = pd.DataFrame(data)

# --- 2. Exploratory Data Analysis ---
print("Plotting Sensor Degradation...")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df[df['MachineID'] == 'MAC-001'], x='Day', y='Vibration', label='Vibration (MAC-001)')
plt.title("Sensor Vibration over Time (Degrading Machine)")
plt.savefig('vibration_trend.png')

# --- 3. Anomaly Detection (Isolation Forest) ---
features = ['Temperature', 'Vibration', 'Pressure', 'RPM']
X = df[features]
y = df['RUL']

print("Training Isolation Forest...")
iso = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = iso.fit_predict(X)

# --- 4. RUL Prediction (XGBoost) ---
print("Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
xgb_model.fit(X, y)

df['Predicted_RUL'] = xgb_model.predict(X)

# --- 5. Model Explainability (SHAP) ---
print("Calculating SHAP Values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X.sample(1000, random_state=42))

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, show=False)
plt.savefig('shap_beeswarm.png')

print("Pipeline completed successfully! Models are trained and explanations are saved.")
