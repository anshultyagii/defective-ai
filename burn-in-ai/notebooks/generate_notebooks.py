import nbformat as nbf
import os

def create_mosfet_anomaly_nb():
    nb = nbf.v4.new_notebook()
    
    text1 = """# MOSFET Anomaly Detection Evaluation
This notebook evaluates the Isolation Forest and Local Outlier Factor (LOF) models trained on the NASA MOSFET dataset."""
    
    code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
import joblib

# Load processed features
df = pd.read_csv('../data/processed/mosfet_features.csv')
features = ['rds_on_norm', 'rds_drift_rate', 'rds_drift_accel', 'peer_deviation', 'temp']
df = df.dropna(subset=features)
"""

    code2 = """# Load trained models
scaler = joblib.load('../models/mosfet/anomaly_scaler.pkl')
iso = joblib.load('../models/mosfet/isolation_forest.pkl')
lof = joblib.load('../models/mosfet/lof.pkl')

X = scaler.transform(df[features])

# Predict anomalies (-1 for anomaly, 1 for normal)
# For Isolation Forest
iso_preds = iso.predict(X)
iso_scores = iso.decision_function(X)

# Since we don't have explicit ground truth labels for "anomaly" in run-to-failure data prior to failure,
# we define proxy labels: components in the top 5% of degradation or with extreme drift as "true anomalies"
true_anomalies = (df['rds_on_norm'] > df['rds_on_norm'].quantile(0.95)) | (df['peer_deviation'] > df['peer_deviation'].quantile(0.95))
y_true = np.where(true_anomalies, -1, 1)

df['anomaly_pred'] = iso_preds
df['true_anomaly'] = y_true
"""

    code3 = """# Evaluate Anomaly Detection
print("Isolation Forest Classification Report:")
print(classification_report(y_true, iso_preds))

# Confusion Matrix
cm = confusion_matrix(y_true, iso_preds, labels=[-1, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly (-1)', 'Normal (1)'], yticklabels=['Anomaly (-1)', 'Normal (1)'])
plt.title('Isolation Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# False Negative Rate (FNR) is critical
# FNR = FN / (FN + TP)
# In our labels, Anomaly is -1.
fn = cm[0, 1]
tp = cm[0, 0]
fnr = fn / (fn + tp)
print(f"False Negative Rate (Missed Defects): {fnr:.4f}")
"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(text1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_code_cell(code3)
    ]
    
    with open('MOSFET_Anomaly_Evaluation.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_mosfet_drift_nb():
    nb = nbf.v4.new_notebook()
    
    text1 = """# MOSFET Drift Prediction Evaluation
This notebook evaluates the regression models for predicting future RDS(on) degradation trajectories."""
    
    code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load processed features
df = pd.read_csv('../data/processed/mosfet_features.csv')
features = ['time_hours', 'rds_on_norm', 'rds_drift_rate', 'rds_drift_accel', 'temp']
"""

    code2 = """# Re-create future targets for evaluation
future_hours = 24
df = df.sort_values(['device', 'time_hours']).copy()
records = []
for device, group in df.groupby('device'):
    times = group['time_hours'].values
    rds_norm = group['rds_on_norm'].values
    for i in range(len(group)):
        target_time = times[i] + future_hours
        if target_time <= times[-1]:
            target_rds = np.interp(target_time, times, rds_norm)
            records.append({
                'device': device,
                'time_hours': times[i],
                'rds_on_norm': rds_norm[i],
                'rds_drift_rate': group['rds_drift_rate'].values[i],
                'rds_drift_accel': group['rds_drift_accel'].values[i],
                'temp': group['temp'].values[i],
                'target_rds_on_norm': target_rds
            })
df_targets = pd.DataFrame(records)

X = df_targets[features]
y = df_targets['target_rds_on_norm']
groups = df_targets['device']

# Group-based split (same as training to get the test set)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
"""

    code3 = """# Load best model
scaler = joblib.load('../models/mosfet/drift_scaler.pkl')
model = joblib.load('../models/mosfet/drift_model.pkl')

X_test_scaled = scaler.transform(X_test)
preds = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2: {r2:.4f}")

plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Future Normalized RDS(on)')
plt.ylabel('Predicted Future Normalized RDS(on)')
plt.title('Drift Prediction: True vs Predicted (Unseen Test Devices)')
plt.show()
"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(text1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_code_cell(code3)
    ]
    
    with open('MOSFET_Drift_Evaluation.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_mosfet_anomaly_nb()
    create_mosfet_drift_nb()
