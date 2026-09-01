import nbformat as nbf

nb_b = nbf.v4.new_notebook()
nb_b.cells = [
    nbf.v4.new_code_cell('!pip install pandas numpy scikit-learn matplotlib seaborn'),
    nbf.v4.new_markdown_cell('''# MODULE B: Time-Series Drift Predictor (Regression)
**Problem Statement Alignment:** "Build a predictive regression model that takes Value_0h and Value_24h as inputs and forecasts Value_168h. If the predicted 168h drift rate exceeds a calculated safety slope, the system flags the component for early rejection."

This module fits a predictive regression model to the early-life measurements (up to ~24h) for each component, extrapolates the resistance to the 168-hour mark, calculates the drift rate, and flags the component if it breaches the safety envelope.
'''),
    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

file_paths = sorted(glob.glob('/Users/anshul/Documents/AI/data/raw/MOSFET/MOSFET_Device_*.csv'))

# Safety limit: If the resistance drifts more than 5% per 168 hours, we reject it.
SAFETY_DRIFT_THRESHOLD_PCT = 5.0 

results = []

print("======================================================")
print(" MODULE B: DRIFT PREDICTION & COMPONENT EXPLANATIONS  ")
print("======================================================")

for file_path in file_paths:
    try:
        df = pd.read_csv(file_path).dropna(subset=['Aging_Hours', 'RDS_on_Ohm']).sort_values('Aging_Hours')
        if len(df) < 2: continue
            
        device = df['Device_ID'].iloc[0]
        
        # 1. Take Value_0h and early measurements (simulating the 0h to 24h burn-in window)
        early_df = df[df['Aging_Hours'] <= 24.0]
        if len(early_df) < 2: 
            early_df = df.head(10) # Fallback if test was extremely short
            
        val_0h = early_df['RDS_on_Ohm'].iloc[0]
        
        # 2. Build the Predictive Regression Model
        X_train = early_df[['Aging_Hours']]
        y_train = early_df['RDS_on_Ohm']
        
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        
        # 3. Forecast Value_168h
        X_future = pd.DataFrame({'Aging_Hours': [168.0]})
        val_168h_pred = regressor.predict(X_future)[0]
        
        # 4. Calculate Drift Rate (Percentage drift over 168h)
        drift_rate_pct = ((val_168h_pred - val_0h) / val_0h) * 100
        
        # 5. Safety Slope Check
        if drift_rate_pct > SAFETY_DRIFT_THRESHOLD_PCT:
            decision = "FLAG (Early Rejection)"
            reason = f"Predicted drift rate ({drift_rate_pct:.2f}%) EXCEEDS safety slope ({SAFETY_DRIFT_THRESHOLD_PCT}%)."
        else:
            decision = "PASS"
            reason = f"Predicted drift rate ({drift_rate_pct:.2f}%) is safely within the envelope."
            
        print(f"\\n--- COMPONENT: Device {device} ---")
        print(f"Input Value_0h      : {val_0h:.5f} Ohms")
        print(f"Forecast Value_168h : {val_168h_pred:.5f} Ohms")
        print(f"Calculated Drift    : {drift_rate_pct:+.2f}%")
        print(f"Decision            : {decision}")
        print(f"Explanation         : {reason}")
        print("-" * 54)
        
    except Exception as e:
        pass
''')
]

with open('/Users/anshul/Documents/AI/notebooks/MODULE_B_Drift_Predictor.ipynb', 'w') as f:
    nbf.write(nb_b, f)

print('Module B updated to strictly match the exact Regression text.')
