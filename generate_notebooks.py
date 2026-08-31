import nbformat as nbf
import os

def create_notebook(filename, cells_content):
    nb = nbf.v4.new_notebook()
    cells = []
    for cell_type, content in cells_content:
        if cell_type == 'md':
            cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == 'code':
            cells.append(nbf.v4.new_code_cell(content))
    nb.cells = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

os.makedirs('notebooks', exist_ok=True)

# Notebook 1: Data Loading, Schema, Validation & Quality
nb1 = [
    ('md', '# Phase 1-4: Data Loading, Schema, Validation & Quality\nThis notebook handles loading data, understanding the schema, validating parameters and checking data quality for MOSFET_Device_01.'),
    ('code', '''import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('data/raw/MOSFET/MOSFET_Device_01.csv')
print(f"Loaded dataset with shape: {df.shape}")
print("\\nSchema:")
print(df.info())

# Report discrepancy based on prompt instructions
print("\\n--- DISCREPANCY REPORT ---")
print("The prompt asked to predict 168h drift based on 0h and 24h for 42 components.")
print(f"However, this file only contains Device_ID: {df['Device_ID'].unique()}")
print(f"Number of unique runs: {df['Run_ID'].nunique()}")
print(f"Max Aging Hours: {df['Aging_Hours'].max():.2f}")
print("As instructed: 'If the dataset contradicts an assumption in this prompt, trust the dataset and explicitly report the discrepancy.'")
print("We will adapt the time windows (e.g., predict max available hour based on earlier hours) and use Run_ID as the grouping variable for demonstration.")
'''),
    ('code', '''# Validate Parameters & Data Quality
print("Missing values:")
print(df.isnull().sum().sum())
print("\\nSummary statistics:")
print(df.describe())
''')
]
create_notebook('notebooks/01_data_loading_and_quality.ipynb', nb1)

# Notebook 2: EDA, Feature Engineering & Target Validation
nb2 = [
    ('md', '# Phase 5-8: EDA, Feature Engineering & Target Validation'),
    ('code', '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/MOSFET/MOSFET_Device_01.csv')

# EDA
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Aging_Hours', y='RDS_on_Ohm', hue='Run_ID', palette='tab10')
plt.title('RDS_on_Ohm over Time per Run')
plt.show()

# Feature Engineering
# Since we don't have 168 hours, we'll extract features for early hours (e.g., 0 to 2) and predict the last hour (e.g., 16).
features = []
for run_id, group in df.groupby('Run_ID'):
    group = group.sort_values('Aging_Hours')
    
    val_0h = group.iloc[0]['RDS_on_Ohm']
    
    # closest to 2 hours
    idx_2h = (group['Aging_Hours'] - 2.0).abs().idxmin()
    val_2h = group.loc[idx_2h, 'RDS_on_Ohm']
    
    # target: last available value (approx 16h)
    val_target = group.iloc[-1]['RDS_on_Ohm']
    
    delta = val_2h - val_0h
    slope = delta / 2.0
    
    features.append({
        'Run_ID': run_id,
        'Value_0h': val_0h,
        'Value_2h': val_2h,
        'Delta_Value': delta,
        'Early_Slope': slope,
        'Target_Value': val_target
    })

feat_df = pd.DataFrame(features)
print(feat_df)
feat_df.to_csv('data/raw/MOSFET/engineered_features.csv', index=False)
''')
]
create_notebook('notebooks/02_eda_and_features.ipynb', nb2)

# Notebook 3: ML Models, Tuning, Evaluation
nb3 = [
    ('md', '# Phase 9-12: ML Models, Tuning & Comparison'),
    ('code', '''import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

feat_df = pd.read_csv('data/raw/MOSFET/engineered_features.csv')

X = feat_df[['Value_0h', 'Value_2h', 'Delta_Value', 'Early_Slope']]
y = feat_df['Target_Value']
groups = feat_df['Run_ID']

# Since we only have 2 runs, KFold with n_splits=2
gkf = GroupKFold(n_splits=2)

models = {
    'Persistence (Baseline)': None,
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

results = []

for name, model in models.items():
    cv_mae = []
    cv_rmse = []
    cv_r2 = []
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if name == 'Persistence (Baseline)':
            # Predict the 2h value
            preds = X_test['Value_2h']
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        cv_mae.append(mean_absolute_error(y_test, preds))
        cv_rmse.append(np.sqrt(mean_squared_error(y_test, preds)))
        # R2 might be NaN for single sample, handle it gracefully
        if len(y_test) > 1:
            cv_r2.append(r2_score(y_test, preds))
        else:
            cv_r2.append(np.nan)
            
    results.append({
        'Model': name,
        'CV MAE': np.mean(cv_mae),
        'CV RMSE': np.mean(cv_rmse),
        'CV R²': np.nanmean(cv_r2) if not np.isnan(cv_r2).all() else np.nan,
        'Std MAE': np.std(cv_mae)
    })

res_df = pd.DataFrame(results)
print(res_df)

# Hyperparameter Tuning with Optuna (Example for XGBoost)
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
    }
    model = XGBRegressor(**params, random_state=42)
    mae_list = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        mae_list.append(mean_absolute_error(y.iloc[test_idx], preds))
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
print("Best Params:", study.best_params)
''')
]
create_notebook('notebooks/03_models_and_tuning.ipynb', nb3)

# Notebook 4: Error Analysis, Explainability, Safety, Small Data Check
nb4 = [
    ('md', '# Phase 13-19: Error Analysis, Explainability, Safety Envelope & Architecture'),
    ('code', '''import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor

feat_df = pd.read_csv('data/raw/MOSFET/engineered_features.csv')
X = feat_df[['Value_0h', 'Value_2h', 'Delta_Value', 'Early_Slope']]
y = feat_df['Target_Value']

# Train final model on all data for explainability (due to extreme small size)
model = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
model.fit(X, y)

feat_df['Predicted_Future_Value'] = model.predict(X)
feat_df['Absolute_Error'] = (feat_df['Target_Value'] - feat_df['Predicted_Future_Value']).abs()
print("Error Analysis Table:")
print(feat_df[['Run_ID', 'Target_Value', 'Predicted_Future_Value', 'Absolute_Error']])

# Explainability (SHAP)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, show=False)

# Phase 18 - Small Dataset Check
print("\\n--- SMALL DATASET CHECK ---")
print(f"Number of components: 1 (Device_ID: 1)")
print(f"Number of trajectories: {len(feat_df)}")
print("Is data sufficient for deep learning? No. Neural networks are not appropriate for 2 samples.")
print("Even tree models and cross-validation are mathematically strained here. Simple baselines are more defensible.")
''')
]
create_notebook('notebooks/04_error_explain_safety.ipynb', nb4)
