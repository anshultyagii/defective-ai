import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

def create_future_targets(df, future_hours=24):
    df = df.sort_values(['device', 'time_hours']).copy()
    records = []
    for device, group in df.groupby('device'):
        times = group['time_hours'].values
        rds_norm = group['rds_on_norm'].values
        for i in range(len(group)):
            current_time = times[i]
            target_time = current_time + future_hours
            if target_time <= times[-1]:
                target_rds = np.interp(target_time, times, rds_norm)
                records.append({
                    'device': device,
                    'time_hours': current_time,
                    'rds_on': group['rds_on'].values[i],
                    'rds_on_norm': rds_norm[i],
                    'rds_drift_rate': group['rds_drift_rate'].values[i],
                    'rds_drift_accel': group['rds_drift_accel'].values[i],
                    'temp': group['temp'].values[i],
                    'target_rds_on_norm': target_rds
                })
    return pd.DataFrame(records)

def train_mosfet_drift(features_file, model_dir):
    df = pd.read_csv(features_file)
    os.makedirs(model_dir, exist_ok=True)
    
    df_targets = create_future_targets(df, future_hours=24)
    if df_targets.empty:
        print("No valid target data.")
        return
        
    features = ['time_hours', 'rds_on_norm', 'rds_drift_rate', 'rds_drift_accel', 'temp']
    X = df_targets[features]
    y = df_targets['target_rds_on_norm']
    groups = df_targets['device']
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=42),
        'LightGBM': HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=43),
    }
    
    best_model_name = None
    best_rmse = float('inf')
    best_model = None
    
    metrics = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            metrics[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model = model
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    print(f"Best model: {best_model_name}")
    
    joblib.dump(scaler, os.path.join(model_dir, 'drift_scaler.pkl'))
    joblib.dump(best_model, os.path.join(model_dir, 'drift_model.pkl'))
    
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'component_type': 'MOSFET',
            'model_version': 'MOSFET-v1.0',
            'features': features,
            'best_model': best_model_name,
            'metrics': metrics
        }, f, indent=4)
        
if __name__ == '__main__':
    train_mosfet_drift('../../../../data/processed/mosfet_features.csv', '../../../../models/mosfet/')
