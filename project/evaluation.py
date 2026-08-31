import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(models, baseline_fn, X, y, groups, cv):
    results = []
    
    # Baseline
    cv_mae = []
    cv_rmse = []
    cv_r2 = []
    
    for train_idx, test_idx in cv.split(X, y, groups):
        preds = baseline_fn(X.iloc[test_idx])
        y_true = y.iloc[test_idx]
        cv_mae.append(mean_absolute_error(y_true, preds))
        cv_rmse.append(np.sqrt(mean_squared_error(y_true, preds)))
        cv_r2.append(r2_score(y_true, preds) if len(y_true) > 1 else np.nan)
        
    results.append({
        'Model': 'Persistence', 
        'CV MAE': np.mean(cv_mae),
        'CV RMSE': np.mean(cv_rmse),
        'CV R²': np.nanmean(cv_r2) if not np.isnan(cv_r2).all() else np.nan,
        'Std MAE': np.std(cv_mae)
    })
    
    # ML Models
    for name, model in models.items():
        cv_mae = []
        cv_rmse = []
        cv_r2 = []
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            y_true = y.iloc[test_idx]
            cv_mae.append(mean_absolute_error(y_true, preds))
            cv_rmse.append(np.sqrt(mean_squared_error(y_true, preds)))
            cv_r2.append(r2_score(y_true, preds) if len(y_true) > 1 else np.nan)
            
        results.append({
            'Model': name, 
            'CV MAE': np.mean(cv_mae),
            'CV RMSE': np.mean(cv_rmse),
            'CV R²': np.nanmean(cv_r2) if not np.isnan(cv_r2).all() else np.nan,
            'Std MAE': np.std(cv_mae)
        })
        
    return pd.DataFrame(results)
