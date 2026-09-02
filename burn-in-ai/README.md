# AI-Driven Dynamic Burn-In Screening & Latent Defect Detection Platform

## Core Differentiator
**Traditional ESS:** Measurement -> Static Limit -> PASS / FAIL
**Proposed AI-Driven ESS:** Measurement History -> Component-Specific Analysis -> Peer/Lot Comparison -> Dynamic Anomaly Detection -> Temporal Drift Analysis -> Future Degradation Prediction -> Safety-Trajectory Analysis -> Component Risk -> Cross-Component Risk Fusion -> Explainable Decision -> NORMAL / WATCH / REJECT

## Architecture

This platform provides separate intelligence for different component physics and unified decision-making for high-reliability screening.
- **Component-Specific ML Pipelines:** MOSFET, IGBT, and Capacitor each run through independent feature engineering, anomaly detection, and drift prediction models.
- **Unified Risk Fusion:** The independent predictions are normalized and fused into a single decision risk score (NORMAL/WATCH/REJECT).
- **Explainability:** SHAP-based natural-language explanation engine to provide insights into why a component was flagged.

## Features Built
1. NASA Datasets successfully loaded, parsed, and profiled (MOSFET, IGBT, Capacitor).
2. End-to-end MOSFET pipeline built (Data extraction -> Normalized $R_{DS(on)}$ features -> Isolation Forest Anomaly Detection -> HistGradientBoosting/Random Forest Drift Prediction).
3. Evaluated Group-based Test Splitting to prevent temporal leakage for multiple components.
4. Dummy models established for IGBT and CAPACITOR to validate the multi-component fusion router (preserving distinct component features).
5. FastAPI Backend running cross-component scoring, inference, and SHAP explainability mockups.
6. React/Vite/Tailwind Frontend demonstrating the Aerospace Mission Control dashboard and a Live Burn-In Simulation for a synthetic component.
7. Fully Dockerized architecture.

## Run Local Environment
```bash
# Start backend
source venv/bin/activate
python backend/run.py

# Start frontend
cd frontend
npm run dev
```

## Run with Docker
```bash
docker-compose up --build
```
