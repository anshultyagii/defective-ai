import os

os.makedirs('project', exist_ok=True)

files = {
    'data_loader.py': '''import pandas as pd
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df
''',
    'data_validation.py': '''def validate_schema(df):
    assert 'Device_ID' in df.columns
    assert 'Run_ID' in df.columns
    assert 'Aging_Hours' in df.columns
    assert 'RDS_on_Ohm' in df.columns
    print("Schema validated.")
''',
    'eda.py': '''import matplotlib.pyplot as plt
import seaborn as sns
def plot_degradation(df):
    sns.lineplot(data=df, x='Aging_Hours', y='RDS_on_Ohm', hue='Run_ID')
    plt.title('Degradation over Time')
    plt.savefig('project/eda_plot.png')
    plt.close()
''',
    'feature_engineering.py': '''import pandas as pd
def engineer_features(df):
    features = []
    for run_id, group in df.groupby('Run_ID'):
        group = group.sort_values('Aging_Hours')
        val_0h = group.iloc[0]['RDS_on_Ohm']
        idx_2h = (group['Aging_Hours'] - 2.0).abs().idxmin()
        val_2h = group.loc[idx_2h, 'RDS_on_Ohm']
        val_target = group.iloc[-1]['RDS_on_Ohm']
        features.append({
            'Run_ID': run_id,
            'Value_0h': val_0h,
            'Value_2h': val_2h,
            'Delta_Value': val_2h - val_0h,
            'Early_Slope': (val_2h - val_0h) / 2.0,
            'Target_Value': val_target
        })
    return pd.DataFrame(features)
''',
    'anomaly_detection.py': '''def detect_anomalies(df):
    # Dynamic outlier detection placeholder
    mean_val = df['Value_0h'].mean()
    std_val = df['Value_0h'].std()
    df['Anomaly_Flag'] = (df['Value_0h'] > mean_val + 3*std_val).astype(int)
    return df
''',
    'baselines.py': '''import numpy as np
def persistence_baseline(X_test):
    return X_test['Value_2h'].values
''',
    'models.py': '''from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

def get_models():
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
''',
    'validation.py': '''from sklearn.model_selection import GroupKFold
def get_cv():
    return GroupKFold(n_splits=2)
''',
    'tuning.py': '''import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def tune_xgboost(X, y, groups, cv):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 4)
        }
        model = XGBRegressor(**params, random_state=42)
        maes = []
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y.iloc[test_idx], preds))
        return np.mean(maes)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    return study.best_params
''',
    'evaluation.py': '''import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(models, baseline_fn, X, y, groups, cv):
    results = []
    
    # Baseline
    cv_mae = []
    for train_idx, test_idx in cv.split(X, y, groups):
        preds = baseline_fn(X.iloc[test_idx])
        cv_mae.append(mean_absolute_error(y.iloc[test_idx], preds))
    results.append({'Model': 'Persistence', 'CV MAE': np.mean(cv_mae)})
    
    # ML Models
    for name, model in models.items():
        cv_mae = []
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            cv_mae.append(mean_absolute_error(y.iloc[test_idx], preds))
        results.append({'Model': name, 'CV MAE': np.mean(cv_mae)})
        
    return pd.DataFrame(results)
''',
    'explainability.py': '''import shap
import matplotlib.pyplot as plt
def explain_model(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('project/shap_summary.png')
    plt.close()
''',
    'safety_envelope.py': '''def apply_safety_envelope(df, predicted_col, boundary):
    df['Flag'] = (df[predicted_col] > boundary).astype(int)
    return df
''',
    'uncertainty.py': '''def conformal_prediction(preds, residuals, alpha=0.1):
    # Dummy conformal prediction
    q = np.quantile(np.abs(residuals), 1 - alpha)
    return preds - q, preds + q
''',
    'main.py': '''import pandas as pd
from data_loader import load_data
from data_validation import validate_schema
from feature_engineering import engineer_features
from evaluation import evaluate_models
from models import get_models
from baselines import persistence_baseline
from validation import get_cv

def main():
    print("Starting pipeline for MOSFET Device 01...")
    df = load_data('../data/raw/MOSFET/MOSFET_Device_01.csv')
    validate_schema(df)
    
    # As noted, only 1 device, 2 runs. Extracting modified features instead of 168h.
    feat_df = engineer_features(df)
    
    X = feat_df[['Value_0h', 'Value_2h', 'Delta_Value', 'Early_Slope']]
    y = feat_df['Target_Value']
    groups = feat_df['Run_ID']
    
    cv = get_cv()
    models = get_models()
    
    res = evaluate_models(models, persistence_baseline, X, y, groups, cv)
    print("\\nModel Comparison:")
    print(res)

if __name__ == '__main__':
    main()
'''
}

for filename, content in files.items():
    with open(f"project/{filename}", 'w') as f:
        f.write(content)
print("Project structure created successfully.")
