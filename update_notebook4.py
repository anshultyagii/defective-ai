import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell('# Phase 13: Error Analysis\nIdentify largest absolute prediction errors, systematic over/underprediction, false-safe predictions.'),
    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

# Load features
feat_df = pd.read_csv('data/raw/MOSFET/engineered_features.csv')
X = feat_df[['Value_0h', 'Value_2h', 'Delta_Value', 'Early_Slope']]
y = feat_df['Target_Value']

# Train final model on all data (since we only have 2 runs, this is just for explainability/demonstration)
model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
model.fit(X, y)

feat_df['Predicted_Future_Value'] = model.predict(X)
feat_df['Absolute_Error'] = (feat_df['Target_Value'] - feat_df['Predicted_Future_Value']).abs()
feat_df['Relative_Error'] = feat_df['Absolute_Error'] / feat_df['Target_Value']

# Define safety boundary for False-Safe analysis (dummy boundary for demonstration)
safety_boundary = feat_df['Target_Value'].mean() + feat_df['Target_Value'].std()
feat_df['Actual_Unsafe'] = feat_df['Target_Value'] > safety_boundary
feat_df['Predicted_Safe'] = feat_df['Predicted_Future_Value'] <= safety_boundary
feat_df['False_Safe'] = feat_df['Actual_Unsafe'] & feat_df['Predicted_Safe']
feat_df['Flag'] = np.where(feat_df['Predicted_Future_Value'] > safety_boundary, 'FLAGGED', 'SAFE')

print("Error Analysis Table:")
display_cols = ['Run_ID', 'Target_Value', 'Predicted_Future_Value', 'Absolute_Error', 'Relative_Error', 'Early_Slope', 'Flag']
print(feat_df[display_cols])

print("\\nFalse-Safe Predictions:")
print(feat_df[feat_df['False_Safe']][display_cols])
'''),
    nbf.v4.new_markdown_cell('# Phase 16: Explainability\nUse SHAP to explain features.'),
    nbf.v4.new_code_cell('''# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("\\n--- GLOBAL EXPLANATION ---")
shap.summary_plot(shap_values, X, show=False)

print("\\n--- INDIVIDUAL EXPLANATION (Run_ID 1) ---")
idx = 0
print(f"FLAGGED / SAFE: {feat_df.iloc[idx]['Flag']}")
print("Primary reasons for prediction:")
for feature_name, shap_val, feat_val in zip(X.columns, shap_values[idx], X.iloc[idx]):
    print(f"{feature_name}: Value={feat_val:.4f}, Contribution={shap_val:.4f}")
print(f"Predicted future value: {feat_df.iloc[idx]['Predicted_Future_Value']:.4f}")
print(f"Safety boundary: {safety_boundary:.4f}")
'''),
    nbf.v4.new_markdown_cell('# Phase 17: Uncertainty\nImplement Conformal Prediction.'),
    nbf.v4.new_code_cell('''# Conformal prediction dummy demonstration
print("\\n--- UNCERTAINTY ESTIMATION ---")
if len(feat_df) < 10:
    print("Warning: Conformal prediction is statistically unreliable because of the extremely small dataset (only 2 samples).")

# Simple residual-based conformal prediction
residuals = np.abs(feat_df['Target_Value'] - feat_df['Predicted_Future_Value'])
alpha = 0.1
q = np.quantile(residuals, 1 - alpha) if len(residuals) > 0 else 0

for i, row in feat_df.iterrows():
    pred = row['Predicted_Future_Value']
    print(f"Run_ID {row['Run_ID']} -> Predicted Value = {pred:.2f}, Prediction Interval = [{pred - q:.2f}, {pred + q:.2f}]")
'''),
    nbf.v4.new_markdown_cell('# Phase 18: Small-Dataset Check'),
    nbf.v4.new_code_cell('''print("\\n--- SMALL DATASET CHECK ---")
print(f"Number of components: {feat_df['Run_ID'].nunique()} (Note: Only 1 device, 2 runs)")
print(f"Number of trajectories: {len(feat_df)}")
print(f"Number of usable training/validation samples: {len(feat_df)}")
print("\\nIs available data sufficient for:")
print("- Random Forest / Gradient Boosting: NO, severely lacks samples for generalization.")
print("- XGBoost: NO")
print("- per-component models: NO, only 2 trajectories.")
print("- uncertainty estimation: NO, statistical methods like conformal prediction require larger calibration sets.")
print("- deep learning: NO, absolutely insufficient data.")
''')
]

nb.cells = cells
with open('notebooks/04_error_explain_safety.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Updated Notebook 4.")
