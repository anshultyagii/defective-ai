import numpy as np
from sklearn.linear_model import LinearRegression
import warnings; warnings.filterwarnings('ignore')

print("==============================================")
print(" 🚀 LIVE DEMO: MODULE B DRIFT PREDICTOR")
print("==============================================\\n")

try:
    # 1. Take User Input
    val_0h = float(input("Enter Component Resistance at 0 Hours (e.g. 0.85): "))
    val_24h = float(input("Enter Component Resistance at 24 Hours (e.g. 0.87): "))
    
    # 2. Format Data for the ML Model
    # X is Aging_Hours, y is Resistance
    X_train = np.array([[0.0], [24.0]])
    y_train = np.array([val_0h, val_24h])
    
    # 3. Train the Regression Model on Live Data
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # 4. Forecast 168 hours
    X_future = np.array([[168.0]])
    val_168h_pred = regressor.predict(X_future)[0]
    
    # 5. Calculate Safety & Decision
    SAFETY_DRIFT_THRESHOLD_PCT = 5.0
    drift_rate_pct = ((val_168h_pred - val_0h) / val_0h) * 100
    
    print("\\n----------------------------------------------")
    print(f"📊 FORECAST RESULTS:")
    print(f"Input 0h         : {val_0h:.4f} Ohms")
    print(f"Input 24h        : {val_24h:.4f} Ohms")
    print(f"Forecast 168h    : {val_168h_pred:.4f} Ohms")
    print(f"Calculated Drift : {drift_rate_pct:+.2f}%")
    
    if drift_rate_pct > SAFETY_DRIFT_THRESHOLD_PCT:
        print("\\n❌ DECISION: FLAG (Early Rejection)")
        print(f"Reason: The projected drift ({drift_rate_pct:.2f}%) exceeds the {SAFETY_DRIFT_THRESHOLD_PCT}% safety limit.")
    else:
        print("\\n✅ DECISION: PASS (Safe)")
        print(f"Reason: The projected drift ({drift_rate_pct:.2f}%) is within safe limits.")
        
except ValueError:
    print("Invalid input. Please enter numerical values.")
