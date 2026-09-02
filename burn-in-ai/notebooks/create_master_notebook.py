import nbformat as nbf
import os

def create_master():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- INTRODUCTION ---
    cells.append(nbf.v4.new_markdown_cell("# AI-Driven Dynamic Burn-In Screening\n\nThis master notebook executes the three independent data pipelines (MOSFET, IGBT, Capacitor), isolates component-level features, evaluates anomalies and 168h drift, and runs the hierarchical risk fusion."))
    
    # --- MOSFET ---
    cells.append(nbf.v4.new_markdown_cell("## 1. MOSFET Pipeline\n\nComponent extraction, time alignment (Equivalent Burn-In Hours), and Random Forest evaluations."))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('../data/processed/mosfet_screening_dataset.csv')
print(f"Total Physical MOSFETs: {df['component_id'].nunique()}")

# Anomaly evaluation
test_df = df.iloc[int(len(df)*0.7):]
print("--- Peer Anomaly Evaluation (Test Set) ---")
print(test_df[['component_id', 'peer_anomaly_score', 'future_risk_label']].head())

# Drift Evaluation
df_drift = df.dropna(subset=['Value_168h']).copy()
train_drift = df_drift.iloc[:int(len(df_drift)*0.7)]
test_drift = df_drift.iloc[int(len(df_drift)*0.7):]

rf = RandomForestRegressor(random_state=42)
features_drift = ['Value_24h', 'delta_0_24', 'slope_0_24', 'peer_deviation']
rf.fit(train_drift[features_drift], train_drift['Value_168h'])
preds = rf.predict(test_drift[features_drift])

print(f"\\nDrift MAE (168h): {mean_absolute_error(test_drift['Value_168h'], preds):.4f}")
print(f"Drift R2 (168h): {r2_score(test_drift['Value_168h'], preds):.4f}")
"""))
    cells.append(nbf.v4.new_code_cell("""with open('../results/mosfet/fusion_results.json', 'r') as f:
    mosfet_risk = json.load(f)['dataset_risk_score']
print(f"MOSFET Dataset Risk Aggregation: {mosfet_risk:.4f}")
"""))

    # --- IGBT ---
    cells.append(nbf.v4.new_markdown_cell("## 2. IGBT Pipeline\n\nIndependent IGBT structural extraction via `Part` IDs."))
    cells.append(nbf.v4.new_code_cell("""igbt_df = pd.read_csv('../results/igbt/component_results.csv')
print("IGBT Extracted Components:")
display(igbt_df[['component_id', 'Value_0h', 'Value_168h', 'peer_anomaly_score', 'component_risk_score', 'component_status']])
with open('../results/igbt/fusion_results.json', 'r') as f:
    igbt_risk = json.load(f)['dataset_risk_score']
print(f"IGBT Dataset Risk Aggregation: {igbt_risk:.4f}")
"""))

    # --- CAPACITOR ---
    cells.append(nbf.v4.new_markdown_cell("## 3. Capacitor Pipeline\n\nIndependent Capacitor structural extraction (`ES10`, `ES12`, `ES14`)."))
    cells.append(nbf.v4.new_code_cell("""cap_df = pd.read_csv('../results/capacitor/component_results.csv')
print("Capacitor Extracted Components:")
display(cap_df[['component_id', 'Value_0h', 'Value_168h', 'peer_anomaly_score', 'component_risk_score', 'component_status']])
with open('../results/capacitor/fusion_results.json', 'r') as f:
    cap_risk = json.load(f)['dataset_risk_score']
print(f"Capacitor Dataset Risk Aggregation: {cap_risk:.4f}")
"""))

    # --- FINAL FUSION ---
    cells.append(nbf.v4.new_markdown_cell("## 4. Final Hierarchical Fusion\n\nCombining the independent dataset risk scores using Maximum-Risk strategy."))
    cells.append(nbf.v4.new_code_cell("""with open('../results/final/final_fusion_results.json', 'r') as f:
    final_fusion = json.load(f)
    
print(json.dumps(final_fusion, indent=4))
"""))

    # --- LEAKAGE AUDIT ---
    cells.append(nbf.v4.new_markdown_cell("## 5. Leakage & Architecture Audit\n\nVerification that no physical component overlaps train/test sets, and no 168h information enters early predictions."))
    cells.append(nbf.v4.new_code_cell("""with open('../results/final/leakage_audit.json', 'r') as f:
    leakage = json.load(f)
print(json.dumps(leakage, indent=4))
"""))

    nb['cells'] = cells
    
    with open('Master_System_Evaluation.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_master()
