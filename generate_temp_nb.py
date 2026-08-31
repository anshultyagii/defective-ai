import nbformat as nbf

# ----- TEMPORARY 06: Forecasting 168h -----
nb6 = nbf.v4.new_notebook()
nb6.cells = [
    nbf.v4.new_code_cell('!pip install pandas numpy matplotlib scikit-learn'),
    nbf.v4.new_markdown_cell('# 168-Hour Extrapolation (Device 08 - Highest Hours)'),
    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. Load Data for Device 08 (Highest hours in dataset: ~131.9h)
df = pd.read_csv('/Users/anshul/Documents/AI/data/raw/MOSFET/MOSFET_Device_08.csv')
df = df.dropna(subset=['Aging_Hours', 'RDS_on_Ohm']).sort_values('Aging_Hours')

print(f"Loaded {len(df)} valid data points for Device 08.")
print(f"Max observed hour: {df['Aging_Hours'].max():.2f}h")
'''),
    nbf.v4.new_code_cell('''# 2. Fit Time-Series Curve
X_train = df[['Aging_Hours']]
y_train = df['RDS_on_Ohm']
baseline_rds = df['Baseline_RDS_Ohm'].iloc[0]

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
model_poly.fit(X_train, y_train)

print("Models trained successfully on all available hours (0 - 131.9h).")
'''),
    nbf.v4.new_code_cell('''# 3. Forecast 168 Hours
X_future = pd.DataFrame({'Aging_Hours': [168.0]})

pred_168h_linear = model_linear.predict(X_future)[0]
pred_168h_poly = model_poly.predict(X_future)[0]

deg_pct_linear = ((pred_168h_linear - baseline_rds) / baseline_rds) * 100
deg_pct_poly = ((pred_168h_poly - baseline_rds) / baseline_rds) * 100

print("--- 168-HOUR FORECAST ---")
print(f"Baseline RDS_on: {baseline_rds:.4f} Ohm")
print(f"Predicted RDS_on at 168h (Linear): {pred_168h_linear:.4f} Ohm  ({deg_pct_linear:+.2f}%)")
print(f"Predicted RDS_on at 168h (Poly):   {pred_168h_poly:.4f} Ohm  ({deg_pct_poly:+.2f}%)")
'''),
    nbf.v4.new_code_cell('''# 4. Anomaly Flagging
safety_threshold_pct = 5.0
print("--- ANOMALY / BURN-IN SCREENING REPORT ---")
if deg_pct_linear > safety_threshold_pct:
    print(f"STATUS: FLAGGED (REJECT) - Projected degradation of {deg_pct_linear:.2f}% exceeds {safety_threshold_pct}% threshold.")
else:
    print(f"STATUS: PASS - Projected degradation of {deg_pct_linear:.2f}% is within safe limits.")
'''),
    nbf.v4.new_code_cell('''# 5. Visualization of the Extrapolation
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', s=1, alpha=0.1, label='Observed Data (0 - 131.9h)')
X_plot = pd.DataFrame({'Aging_Hours': np.linspace(0, 175, 500)})
plt.plot(X_plot, model_linear.predict(X_plot), color='red', linestyle='--', linewidth=2, label='Linear Forecast')
plt.plot(X_plot, model_poly.predict(X_plot), color='orange', linestyle='--', linewidth=2, label='Polynomial Forecast (Deg 2)')
plt.scatter([168.0], [pred_168h_linear], color='red', s=100, zorder=5, label='168h Linear Prediction')
plt.scatter([168.0], [pred_168h_poly], color='orange', s=100, zorder=5, label='168h Poly Prediction')
plt.title('Extrapolating RDS_on_Ohm to 168 Hours (Device 08)')
plt.xlabel('Aging Hours')
plt.ylabel('RDS_on_Ohm')
plt.legend()
plt.grid(True)
plt.show()
''')
]
with open('/Users/anshul/Documents/AI/notebooks/TEMPORARY_06_forecasting_168h.ipynb', 'w') as f:
    nbf.write(nb6, f)

# ----- TEMPORARY 07: Backtesting -----
nb7 = nbf.v4.new_notebook()
nb7.cells = [
    nbf.v4.new_code_cell('!pip install pandas numpy matplotlib scikit-learn'),
    nbf.v4.new_markdown_cell('# Forecast Verification Backtesting (Device 08)'),
    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_absolute_error

df = pd.read_csv('/Users/anshul/Documents/AI/data/raw/MOSFET/MOSFET_Device_08.csv')
df = df.dropna(subset=['Aging_Hours', 'RDS_on_Ohm']).sort_values('Aging_Hours')

# Cutoff at 100 hours
cutoff_hour = 100.0
train_df = df[df['Aging_Hours'] <= cutoff_hour]
test_df = df[df['Aging_Hours'] > cutoff_hour]

print(f"Training on {len(train_df)} points (0 to 100h).")
print(f"Testing on {len(test_df)} points (100h to 131.9h).")

X_train = train_df[['Aging_Hours']]
y_train = train_df['RDS_on_Ohm']
X_test = test_df[['Aging_Hours']]
y_test = test_df['RDS_on_Ohm']

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"--- REGRESSION ERROR ---")
print(f"Mean Absolute Error (MAE) for future points: {mae:.4f} Ohm")

# Classification Verification (Anomaly Detection)
threshold = y_train.max() * 1.05 # Dynamic threshold for demonstration (5% over max training value)
actual_flags = (y_test > threshold).astype(int)
predicted_flags = (y_pred > threshold).astype(int)

acc = accuracy_score(actual_flags, predicted_flags)
rec = recall_score(actual_flags, predicted_flags, zero_division=0)
prec = precision_score(actual_flags, predicted_flags, zero_division=0)

print(f"\\nUsing dynamic anomaly threshold of {threshold:.4f} Ohm")
print("\\n--- CLASSIFICATION METRICS (Forecast Verification) ---")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
'''),
    nbf.v4.new_code_cell('''plt.figure(figsize=(12, 6))
# Plot 1% of points to not crash browser with massive scatter
plt.scatter(X_train[::100], y_train[::100], color='blue', s=2, label='Training Data (Sampled)')
plt.scatter(X_test[::100], y_test[::100], color='green', s=2, label='Actual Future Data (Sampled)')

X_plot = pd.DataFrame({'Aging_Hours': np.linspace(0, 135, 500)})
plt.plot(X_plot, model.predict(X_plot), color='red', linestyle='--', linewidth=2, label='Model Forecast')

plt.axvline(x=cutoff_hour, color='black', linestyle=':', label='Time Cutoff (100h)')
plt.axhline(y=threshold, color='orange', linestyle='-', alpha=0.5, label='Anomaly Threshold')
plt.title('Backtesting Forecast Accuracy (Device 08)')
plt.xlabel('Aging Hours')
plt.ylabel('RDS_on_Ohm')
plt.legend()
plt.grid(True)
plt.show()
''')
]
with open('/Users/anshul/Documents/AI/notebooks/TEMPORARY_07_forecast_backtesting.ipynb', 'w') as f:
    nbf.write(nb7, f)
print("Temporary notebooks created successfully.")
