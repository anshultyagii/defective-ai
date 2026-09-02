from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import json
import shap

app = FastAPI(title="AI-Driven Dynamic Burn-In Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataProfileRequest(BaseModel):
    component_type: str
    
class PredictRequest(BaseModel):
    features: dict

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data')

@app.get("/")
def read_root():
    return {"status": "AI-Driven Dynamic Burn-In Screening API is running"}

@app.get("/api/models/{component}/metrics")
def get_model_metrics(component: str):
    meta_path = os.path.join(MODEL_DIR, component.lower(), 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model metadata not found")

@app.post("/api/screen/{component}")
def screen_component(component: str, req: PredictRequest):
    comp_dir = os.path.join(MODEL_DIR, component.lower())
    if not os.path.exists(comp_dir):
        raise HTTPException(status_code=404, detail="Component models not found")
        
    try:
        # Load Anomaly Models
        scaler_anom = joblib.load(os.path.join(comp_dir, 'anomaly_scaler.pkl'))
        iso = joblib.load(os.path.join(comp_dir, 'isolation_forest.pkl'))
        lof = joblib.load(os.path.join(comp_dir, 'lof.pkl'))
        stats = joblib.load(os.path.join(comp_dir, 'statistical_bounds.pkl'))
        
        # Load Drift Models
        scaler_drift = joblib.load(os.path.join(comp_dir, 'drift_scaler.pkl'))
        drift_model = joblib.load(os.path.join(comp_dir, 'drift_model.pkl'))
        
        with open(os.path.join(comp_dir, 'metadata.json'), 'r') as f:
            meta = json.load(f)
            
        features = pd.DataFrame([req.features])
        
        # Anomaly scoring
        anom_features = ['rds_on_norm', 'rds_drift_rate', 'rds_drift_accel', 'peer_deviation', 'temp']
        # If any feature is missing, fill with 0
        for col in anom_features:
            if col not in features:
                features[col] = 0.0
                
        X_anom = scaler_anom.transform(features[anom_features])
        iso_score = iso.decision_function(X_anom)[0] # negative is anomalous
        iso_pred = iso.predict(X_anom)[0] # -1 is anomalous
        
        # Normalize anomaly score to 0-1 (1 is highly anomalous)
        anomaly_score = 1.0 - (1.0 / (1.0 + np.exp(-iso_score)))
        
        # Drift prediction
        drift_features = meta['features']
        for col in drift_features:
            if col not in features:
                features[col] = 0.0
                
        X_drift = scaler_drift.transform(features[drift_features])
        predicted_drift = drift_model.predict(X_drift)[0]
        
        # Risk fusion
        safety_margin = 5.0 - predicted_drift # Assuming limit is 5.0 normalized rds
        
        unified_risk = (anomaly_score * 0.5) + (max(0, (predicted_drift - 1.0) / 4.0) * 0.5)
        
        decision = "NORMAL"
        if unified_risk > 0.7 or predicted_drift > 4.5:
            decision = "REJECT"
        elif unified_risk > 0.4 or predicted_drift > 2.0:
            decision = "WATCH"
            
        return {
            "component": component.upper(),
            "model_version": meta.get("model_version", "v1.0"),
            "anomaly_score": round(float(anomaly_score), 4),
            "predicted_future_value_norm": round(float(predicted_drift), 4),
            "safety_margin": round(float(safety_margin), 4),
            "unified_risk": round(float(unified_risk), 4),
            "decision": decision,
            "static_limit": "PASS" # Assuming all pass static limit for demo
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/explain/{component}/{id}")
def explain_decision(component: str, id: str):
    # This is a mock response demonstrating the required SHAP-based explanation
    # In a real system, we would calculate SHAP values for the specific sample `id`
    
    comp_upper = component.upper()
    if comp_upper == 'MOSFET':
        explanation = "This MOSFET was flagged because its normalized ON-state resistance has increased significantly relative to the lot baseline, while its degradation slope indicates a high probability of entering the unsafe trajectory before the end of the burn-in period."
        features = [
            {"name": "rds_drift_rate", "contribution": "HIGH"},
            {"name": "peer_deviation", "contribution": "HIGH"},
            {"name": "rds_drift_accel", "contribution": "MEDIUM"},
            {"name": "temp", "contribution": "LOW"}
        ]
    elif comp_upper == 'IGBT':
        explanation = "This IGBT exhibits elevated VCE degradation and high parameter drift rates. Proceed with caution."
        features = [
            {"name": "vce_norm", "contribution": "HIGH"},
            {"name": "ic_drift", "contribution": "MEDIUM"},
            {"name": "temp", "contribution": "LOW"}
        ]
    elif comp_upper == 'CAPACITOR':
        explanation = "Flagged due to severe ESR increase and capacitance loss."
        features = [
            {"name": "esr_increase_pct", "contribution": "HIGH"},
            {"name": "cap_loss_pct", "contribution": "HIGH"}
        ]
    else:
        explanation = "Unknown component."
        features = []
        
    return {
        "component": comp_upper,
        "id": id,
        "decision": "REJECT",
        "explanation_text": explanation,
        "shap_features": features
    }
