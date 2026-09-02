import os
import glob
import scipy.io
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, fbeta_score, recall_score, precision_score
import shap

# --- SETTINGS & PATHS ---
RAW_DIR = '../../../../data/raw/mosfet'
PROC_DIR = '../../../../data/processed'
RES_DIR = '../../../../results/mosfet'
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs('../../../../models/mosfet_new', exist_ok=True)

# 1. AUDIT & EXTRACTION
def extract_all():
    files = glob.glob(os.path.join(RAW_DIR, "*.mat"))
    devices = sorted(list(set([os.path.basename(f).split('_run_')[0] for f in files if "Test" in f])))
    
    all_data = []
    
    for device in devices:
        dfiles = glob.glob(os.path.join(RAW_DIR, f"{device}_run_*.mat"))
        dfiles = [f for f in dfiles if 'error' not in f]
        dfiles.sort(key=lambda x: int(os.path.basename(x).split('_run_')[1].split('.mat')[0]))
        
        start_time = None
        device_records = []
        for f in dfiles:
            try:
                mat = scipy.io.loadmat(f)
                if 'measurement' not in mat: continue
                meas = mat['measurement'][0,0]
                if 'steadyState' not in meas.dtype.names: continue
                ss = meas['steadyState']
                n_points = ss.shape[1]
                
                # Subsample for speed during extraction (every 100th point)
                for i in range(0, n_points, 100):
                    t = ss[0, i]['timeEpoch'][0,0]
                    vds = ss[0, i]['timeDomain'][0,0]['drainSourceVoltage'][0,0].flatten()
                    id_ = ss[0, i]['timeDomain'][0,0]['drainCurrent'][0,0].flatten()
                    temp = ss[0, i]['timeDomain'][0,0]['packageTemperature'][0,0].flatten()
                    
                    vds_mean = np.mean(vds)
                    id_mean = np.mean(id_)
                    temp_mean = np.mean(temp)
                    rds_on = np.abs(vds_mean / (id_mean + 1e-6))
                    
                    if start_time is None: start_time = t
                    
                    # Accelerated time in hours
                    time_hours = (t - start_time) * 24
                    
                    device_records.append({
                        'device': device,
                        'time_hours': time_hours,
                        'vds': vds_mean,
                        'id': id_mean,
                        'temp': temp_mean,
                        'rds_on': rds_on
                    })
            except:
                pass
        
        if device_records:
            all_data.extend(device_records)
            
    df = pd.DataFrame(all_data)
    return df

print("Extracting data...")
raw_df = extract_all()

# 2. TIME ALIGNMENT & COMPONENT-LEVEL DATASET
# Problem: Actual accelerated time is ~10-20 hours. 
# We need to map to 0h, 24h, 96h, 168h strictly for the burn-in formulation.
# We will use "Equivalent Burn-In Hours" (EBH). We assume max observed time ~ 15 hours maps to 200 EBH.
print("Aligning time...")
device_lifetimes = raw_df.groupby('device')['time_hours'].max()
p20_lifetime = device_lifetimes.quantile(0.2)
# Scale so that 80% of devices survive past 200 EBH
time_scale_factor = 200.0 / p20_lifetime
raw_df['ebh'] = raw_df['time_hours'] * time_scale_factor

def interpolate_checkpoint(device_df, target_ebh):
    times = device_df['ebh'].values
    rds = device_df['rds_on'].values
    if target_ebh <= times[-1]:
        return np.interp(target_ebh, times, rds)
    return np.nan

component_records = []
for device, group in raw_df.groupby('device'):
    group = group.sort_values('ebh')
    
    val_0 = interpolate_checkpoint(group, 0)
    val_24 = interpolate_checkpoint(group, 24)
    val_96 = interpolate_checkpoint(group, 96)
    val_168 = interpolate_checkpoint(group, 168)
    
    if pd.isna(val_0) or pd.isna(val_24):
        continue # Device didn't even survive to 24h EBH
        
    delta_0_24 = val_24 - val_0
    slope_0_24 = delta_0_24 / 24.0
    
    delta_24_96 = (val_96 - val_24) if not pd.isna(val_96) else np.nan
    slope_24_96 = (delta_24_96 / 72.0) if not pd.isna(val_96) else np.nan
    
    # We define a "defect risk" target: if degradation at 168h is extremely high OR device failed before 168h
    # NASA data is run-to-failure. If val_168 is NaN, it failed before 168h -> HIGH RISK (1).
    # If val_168 > 1.5 * val_0 -> HIGH RISK (1).
    if pd.isna(val_168):
        future_risk_label = 1
    elif val_168 > 1.5 * val_0:
        future_risk_label = 1
    else:
        future_risk_label = 0
        
    component_records.append({
        'component_id': device,
        'Value_0h': val_0,
        'Value_24h': val_24,
        'Value_96h': val_96,
        'Value_168h': val_168,
        'delta_0_24': delta_0_24,
        'delta_24_96': delta_24_96,
        'slope_0_24': slope_0_24,
        'slope_24_96': slope_24_96,
        'future_risk_label': future_risk_label
    })

comp_df = pd.DataFrame(component_records)

# Calculate Peer Deviation using TRAINING set only (Leakage prevention)
# We need to split first.
print("Splitting and calculating peer deviation...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(comp_df, groups=comp_df['component_id']))

train_df = comp_df.iloc[train_idx].copy()
test_df = comp_df.iloc[test_idx].copy()

# Robust statistics on training
median_0h = train_df['Value_0h'].median()
mad_0h = np.median(np.abs(train_df['Value_0h'] - median_0h)) + 1e-6

median_24h = train_df['Value_24h'].median()
mad_24h = np.median(np.abs(train_df['Value_24h'] - median_24h)) + 1e-6

train_df['peer_deviation'] = np.abs(train_df['Value_24h'] - median_24h) / mad_24h
test_df['peer_deviation'] = np.abs(test_df['Value_24h'] - median_24h) / mad_24h

final_df = pd.concat([train_df, test_df]).sort_index()
final_df.to_csv(os.path.join(PROC_DIR, 'mosfet_screening_dataset.csv'), index=False)

# 3. ANOMALY DETECTION (UNSUPERVISED)
print("Training anomaly models...")
features_anom = ['Value_24h', 'delta_0_24', 'slope_0_24', 'peer_deviation']

iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(train_df[features_anom])
final_df['peer_anomaly_score'] = -iso.decision_function(final_df[features_anom]) # higher is more anomalous

# 4. SUPERVISED DEFECT-RISK
print("Training supervised models...")
features_sup = ['Value_24h', 'delta_0_24', 'slope_0_24', 'peer_deviation']

scaler = StandardScaler()
X_train_sup = scaler.fit_transform(train_df[features_sup])
X_test_sup = scaler.transform(test_df[features_sup])
y_train_sup = train_df['future_risk_label']
y_test_sup = test_df['future_risk_label']

rf_sup = RandomForestRegressor(n_estimators=100, random_state=42)
rf_sup.fit(X_train_sup, y_train_sup)
test_df['supervised_defect_probability'] = rf_sup.predict(X_test_sup)

# 5. DRIFT PREDICTION (Predict Value_168h)
# We can only train drift on components that actually reached 168h! 
# Or we impute/predict failure time. For drift, we predict Value_168h.
train_drift_df = train_df.dropna(subset=['Value_168h']).copy()
test_drift_df = test_df.dropna(subset=['Value_168h']).copy()

if not train_drift_df.empty and not test_drift_df.empty:
    X_train_drift = scaler.fit_transform(train_drift_df[features_sup])
    X_test_drift = scaler.transform(test_drift_df[features_sup])
    y_train_drift = train_drift_df['Value_168h']
    y_test_drift = test_drift_df['Value_168h']
    
    rf_drift = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_drift.fit(X_train_drift, y_train_drift)
    
    preds = rf_drift.predict(X_test_drift)
    mae = mean_absolute_error(y_test_drift, preds)
    rmse = np.sqrt(mean_squared_error(y_test_drift, preds))
    r2 = r2_score(y_test_drift, preds)
    
    drift_metrics = {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}
    with open(os.path.join(RES_DIR, 'drift_metrics.json'), 'w') as f:
        json.dump(drift_metrics, f, indent=4)
        
    test_drift_df['Predicted_168h'] = preds
    test_drift_df.to_csv(os.path.join(RES_DIR, 'screening_results.csv'), index=False)

print("Pipeline complete.")
