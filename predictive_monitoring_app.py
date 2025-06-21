
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import os

def load_data(path='usage_metrics.csv'):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

def create_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['prev_cpu'] = df['cpu_usage'].shift(1)
    df['prev_mem'] = df['memory_usage'].shift(1)
    df['prev_disk'] = df['disk_usage'].shift(1)
    return df.dropna()

def train_model(df):
    X = df[['hour', 'dayofweek', 'prev_cpu', 'prev_mem', 'prev_disk']]
    y = df['cpu_usage']
    model = RandomForestRegressor()
    model.fit(X, y)
    joblib.dump(model, 'resource_model.pkl')
    return model

def predict_risk(model, metrics):
    X_pred = pd.DataFrame([metrics])
    pred_cpu = model.predict(X_pred)[0]
    risk_score = min(100, max(0, (pred_cpu - 70) * 2))
    return pred_cpu, risk_score

def dashboard():
    st.title("Predictive Resource Monitoring")
    df = create_features(load_data())
    model = joblib.load('resource_model.pkl') if os.path.exists('resource_model.pkl') else train_model(df)
    last = df.iloc[-1]
    metrics = {
        'hour': last['hour'], 'dayofweek': last['dayofweek'],
        'prev_cpu': last['cpu_usage'], 'prev_mem': last['memory_usage'], 'prev_disk': last['disk_usage']
    }
    pred_cpu, risk_score = predict_risk(model, metrics)
    st.metric("Predicted CPU Usage", f"{pred_cpu:.2f}%")
    st.metric("Risk Score", f"{risk_score:.2f}%")
    st.line_chart(df.set_index('timestamp')[['cpu_usage', 'memory_usage', 'disk_usage']])

if __name__ == '__main__':
    dashboard()
