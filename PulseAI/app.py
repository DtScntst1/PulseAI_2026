import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import pickle
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="PulseAI", page_icon="⚙️", layout="wide")

# Custom Cyber-Industrial CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #161B22;
        border-left: 5px solid #00FF41;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,255,65,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #00FF41;
    }
    .metric-label {
        color: #8B949E;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .alert-card {
        background-color: #2D1115;
        border-left: 5px solid #FF4136;
        border-radius: 5px;
        padding: 15px;
        color: #FF4136;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model_and_data():
    try:
        with open('models/anomaly_model.pkl', 'rb') as f:
            iso_forest = pickle.load(f)
        with open('models/rul_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        df_raw = pd.read_csv('data/sensor_data.csv')
        return iso_forest, xgb_model, feature_cols, df_raw
    except Exception:
        # Generate data and train on the fly for robust cloud deployment
        import sys
        sys.path.append('.')
        from src.train_model import train_and_save_model
        train_and_save_model()
        
        with open('models/anomaly_model.pkl', 'rb') as f:
            iso_forest = pickle.load(f)
        with open('models/rul_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        df_raw = pd.read_csv('data/sensor_data.csv')
        return iso_forest, xgb_model, feature_cols, df_raw

# App Header
st.title("⚙️ PulseAI: Industrial Predictive Maintenance")
st.markdown("**Real-time Anomaly Detection & Remaining Useful Life (RUL) Forecasting.**")

with st.spinner("Initializing IoT Sensors & ML Models..."):
    iso_forest, xgb_model, feature_cols, df_raw = get_model_and_data()

# Latest data for each machine
df_latest = df_raw.sort_values('Day').groupby('MachineID').tail(1).copy()
X_latest = df_latest[feature_cols]

# Predictions
df_latest['Anomaly_Score'] = iso_forest.decision_function(X_latest)
df_latest['Is_Anomaly'] = iso_forest.predict(X_latest)
df_latest['Predicted_RUL'] = np.maximum(0, xgb_model.predict(X_latest))

anomalies = df_latest[df_latest['Is_Anomaly'] == -1]

# Tabs
tab1, tab2, tab3 = st.tabs(["🎛️ Live Control Room", "🔬 Diagnostic Center (XAI)", "🏭 Fleet Overview"])

with tab1:
    st.header("Live Control Room")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_latest)}</div><div class="metric-label">Active Machines</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {"#FF4136" if len(anomalies) > 0 else "#00FF41"}">{len(anomalies)}</div><div class="metric-label">Critical Alerts</div></div>', unsafe_allow_html=True)
    with col3:
        avg_rul = df_latest["Predicted_RUL"].mean()
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_rul:.0f} Days</div><div class="metric-label">Avg Fleet RUL</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">$2.4M</div><div class="metric-label">Estimated Savings</div></div>', unsafe_allow_html=True)
        
    st.write("---")
    
    st.subheader("Simulated Real-Time Sensor Feed (Select Machine)")
    selected_machine = st.selectbox("Machine ID", df_latest['MachineID'])
    machine_history = df_raw[df_raw['MachineID'] == selected_machine]
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.write("**Vibration History**")
        st.line_chart(machine_history.set_index('Day')['Vibration'], color="#00FF41")
    with col_g2:
        st.write("**Temperature History**")
        st.line_chart(machine_history.set_index('Day')['Temperature'], color="#FF851B")

with tab2:
    st.header("Diagnostic Center (XAI)")
    st.markdown("Use **SHAP** values to understand exactly *why* a machine's Remaining Useful Life (RUL) is degrading.")
    
    diag_machine = st.selectbox("Select Machine for Root Cause Analysis", df_latest['MachineID'], key='diag')
    mach_data = df_latest[df_latest['MachineID'] == diag_machine][feature_cols]
    mach_rul = df_latest[df_latest['MachineID'] == diag_machine]['Predicted_RUL'].values[0]
    is_anom = df_latest[df_latest['MachineID'] == diag_machine]['Is_Anomaly'].values[0] == -1
    
    if is_anom:
        st.markdown(f"<div class='alert-card'><strong>⚠️ CRITICAL ALERT:</strong> Machine {diag_machine} is exhibiting anomalous behavior!</div><br>", unsafe_allow_html=True)
    
    st.subheader(f"Predicted RUL: **{mach_rul:.0f} Days**")
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(mach_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

with tab3:
    st.header("Fleet Overview")
    
    st.subheader("Fleet Remaining Useful Life (RUL) Distribution")
    rul_hist, bin_edges = np.histogram(df_latest['Predicted_RUL'], bins=15)
    chart_data = pd.DataFrame({
        "RUL (Days)": bin_edges[:-1],
        "Machine Count": rul_hist
    }).set_index("RUL (Days)")
    
    st.bar_chart(chart_data, color="#39CCCC")
    
    st.write("---")
    st.subheader("Machines Requiring Immediate Maintenance")
    critical_df = df_latest[df_latest['Predicted_RUL'] < 30][['MachineID', 'Predicted_RUL', 'Temperature', 'Vibration']].sort_values('Predicted_RUL')
    if len(critical_df) > 0:
        st.dataframe(critical_df, use_container_width=True)
    else:
        st.success("No machines require immediate maintenance.")
