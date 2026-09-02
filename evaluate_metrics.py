import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('burn-in-ai/data/processed/mosfet_screening_dataset.csv')

# --- Anomaly Model Metrics ---
test_df = df.iloc[int(len(df)*0.7):].copy()
threshold = test_df['peer_anomaly_score'].quantile(0.8)
preds = (test_df['peer_anomaly_score'] > threshold).astype(int)
y_true = test_df['future_risk_label']

acc = accuracy_score(y_true, preds)
rec = recall_score(y_true, preds, zero_division=0)
prec = precision_score(y_true, preds, zero_division=0)
cm = confusion_matrix(y_true, preds)

print("--- PEER ANOMALY DETECTION (UNSUPERVISED) ---")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
if cm.shape == (2,2):
    fnr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    print(f"False Negative Rate: {fnr*100:.2f}%")
else:
    print("FNR: N/A (No positives in test set)")

# --- Drift Model Metrics ---
df_drift = df.dropna(subset=['Value_168h']).copy()
train_drift = df_drift.iloc[:int(len(df_drift)*0.7)]
test_drift = df_drift.iloc[int(len(df_drift)*0.7):]

features_drift = ['Value_24h', 'delta_0_24', 'slope_0_24', 'peer_deviation']
rf = RandomForestRegressor(random_state=42)
rf.fit(train_drift[features_drift], train_drift['Value_168h'])
drift_preds = rf.predict(test_drift[features_drift])
y_true_drift = test_drift['Value_168h']

mae = mean_absolute_error(y_true_drift, drift_preds)
r2 = r2_score(y_true_drift, drift_preds)

print("\n--- 168h DRIFT PREDICTION (RANDOM FOREST) ---")
print(f"MAE: {mae:.4f}")
print(f"R2:  {r2:.4f}")
