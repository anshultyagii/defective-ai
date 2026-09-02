import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_mosfet_anomaly(features_file, model_dir):
    df = pd.read_csv(features_file)
    os.makedirs(model_dir, exist_ok=True)
    
    # We select features for anomaly detection
    features = ['rds_on_norm', 'rds_drift_rate', 'rds_drift_accel', 'peer_deviation', 'temp']
    
    # Drop NAs
    df = df.dropna(subset=features).copy()
    
    # We scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    # Train Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X)
    
    # LOF (we only use fit_predict for scoring later, but we can save a novelty LOF)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
    lof.fit(X)
    
    # We define statistical boundaries for 'healthy' based on 95th percentile
    stats = {
        'rds_on_norm_95': df['rds_on_norm'].quantile(0.95),
        'peer_deviation_95': df['peer_deviation'].quantile(0.95),
        'drift_rate_95': df['rds_drift_rate'].quantile(0.95)
    }
    
    joblib.dump(scaler, os.path.join(model_dir, 'anomaly_scaler.pkl'))
    joblib.dump(iso, os.path.join(model_dir, 'isolation_forest.pkl'))
    joblib.dump(lof, os.path.join(model_dir, 'lof.pkl'))
    joblib.dump(stats, os.path.join(model_dir, 'statistical_bounds.pkl'))
    
    print("Anomaly models saved successfully.")

if __name__ == '__main__':
    train_mosfet_anomaly('../../../../data/processed/mosfet_features.csv', '../../../../models/mosfet/')
