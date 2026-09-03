import os
import glob
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, fbeta_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../data'))
RES_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../results'))

def run_kfold_evaluation():
    print("Running GroupKFold OOF Evaluation...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'processed/mosfet_screening_dataset.csv'))
    
    # Drop previous leaked features
    if 'peer_anomaly_score' in df.columns: df = df.drop(columns=['peer_anomaly_score'])
    if 'peer_deviation' in df.columns: df = df.drop(columns=['peer_deviation'])
    
    # Prepare Out-Of-Fold tracking
    df['oof_anomaly_score'] = np.nan
    df['oof_predicted_168h'] = np.nan
    df['anomaly_threshold_used'] = np.nan
    
    features = ['Value_24h', 'delta_0_24', 'slope_0_24', 'peer_deviation']
    
    gkf = GroupKFold(n_splits=5)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df['component_id'])):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Leakage Assertion
        assert len(set(train_df['component_id']).intersection(set(test_df['component_id']))) == 0, "Leakage!"
        
        # Feature Engineering (from TRAIN only)
        train_median = train_df['Value_24h'].median()
        train_mad = (train_df['Value_24h'] - train_median).abs().median()
        if train_mad == 0: train_mad = 1e-5
        
        train_df['peer_deviation'] = (train_df['Value_24h'] - train_median).abs() / train_mad
        test_df['peer_deviation'] = (test_df['Value_24h'] - train_median).abs() / train_mad
        
        train_medians = train_df[features].median()
        train_df[features] = train_df[features].fillna(train_medians)
        test_df[features] = test_df[features].fillna(train_medians)
        
        # Anomaly Detection
        iso = IsolationForest(random_state=42, contamination=0.1)
        iso.fit(train_df[features])
        
        train_scores = -iso.decision_function(train_df[features])
        test_scores = -iso.decision_function(test_df[features])
        
        # We simulate a validation set by simply using a Train quantile (since we can't afford 3-way split on 42 samples easily)
        # We use the 80th percentile of TRAIN as the threshold for TEST
        anomaly_threshold = np.quantile(train_scores, 0.8)
        
        # Drift Regressor (Only train on components that survived to 168h in train set)
        train_drift = train_df.dropna(subset=['Value_168h'])
        rf = RandomForestRegressor(random_state=42, n_estimators=100)
        
        if len(train_drift) > 0:
            rf.fit(train_drift[features], train_drift['Value_168h'])
            test_preds = rf.predict(test_df[features])
        else:
            test_preds = np.full(len(test_df), np.nan)
            
        # Store OOF predictions
        df.loc[test_idx, 'oof_anomaly_score'] = test_scores
        df.loc[test_idx, 'oof_predicted_168h'] = test_preds
        df.loc[test_idx, 'anomaly_threshold_used'] = anomaly_threshold
        
    # Now calculate Component Risk and Status on OOF predictions!
    df['static_status'] = np.where(df['Value_24h'] > df['Value_0h'] * 1.5, 'FAIL', 'PASS') # Unvalidated Engineering Threshold
    df['anomaly_flag'] = np.where(df['oof_anomaly_score'] > df['anomaly_threshold_used'], 1, 0)
    
    df['predicted_total_drift'] = df['oof_predicted_168h'] - df['Value_24h']
    df['safety_margin'] = (df['Value_0h'] * 2.0) - df['oof_predicted_168h'] # Unvalidated projection margin
    
    norm_anomaly = (df['oof_anomaly_score'] - df['oof_anomaly_score'].min()) / (df['oof_anomaly_score'].max() - df['oof_anomaly_score'].min() + 1e-5)
    df['component_risk_score'] = norm_anomaly * 0.5 + (np.where(df['safety_margin'] < 0, 1.0, 0.0)) * 0.5
    df['component_status'] = np.where(df['component_risk_score'] > 0.6, 'REJECT', np.where(df['component_risk_score'] > 0.3, 'WATCH', 'NORMAL'))
    
    reasons = []
    for _, row in df.iterrows():
        r = []
        if row['static_status'] == 'FAIL': r.append('STATIC_LIMIT_VIOLATION')
        if row['anomaly_flag'] == 1: r.append('HIGH_PEER_DEVIATION')
        if row['safety_margin'] < 0: r.append('LOW_PROJECTION_MARGIN')
        reasons.append("|".join(r) if r else 'NONE')
    df['reason_codes'] = reasons
    df['dataset'] = 'MOSFET'
    
    # Save the OOF results
    df.to_csv(os.path.join(RES_DIR, 'mosfet/component_results.csv'), index=False)
    
    # -----------------------------------------------------
    # METRICS EVALUATION ON FULL OOF SET
    # -----------------------------------------------------
    y_true = df['future_risk_label'].fillna(0)
    preds = np.where(df['component_status'].isin(['WATCH', 'REJECT']), 1, 0)
    static_preds = np.where(df['static_status'] == 'FAIL', 1, 0)
    
    metrics = {
        "AI_Accuracy": accuracy_score(y_true, preds),
        "AI_Recall": recall_score(y_true, preds, zero_division=0),
        "AI_Precision": precision_score(y_true, preds, zero_division=0),
        "Static_Accuracy": accuracy_score(y_true, static_preds),
        "Static_Recall": recall_score(y_true, static_preds, zero_division=0),
    }
    
    comp_df = pd.DataFrame([
        {"Method": "Static Baseline", "Recall": metrics['Static_Recall'], "Accuracy": metrics['Static_Accuracy']},
        {"Method": "Dynamic AI Risk", "Recall": metrics['AI_Recall'], "Accuracy": metrics['AI_Accuracy']}
    ])
    comp_df.to_csv(os.path.join(RES_DIR, 'mosfet/static_vs_dynamic.csv'), index=False)
    
    # Save high risk static pass
    high_risk = df[(df['static_status'] == 'PASS') & (df['component_status'].isin(['REJECT', 'WATCH']))]
    high_risk.to_csv(os.path.join(RES_DIR, 'mosfet/high_risk_static_pass.csv'), index=False)
    
    print("OOF Evaluation Complete.")

if __name__ == '__main__':
    run_kfold_evaluation()
