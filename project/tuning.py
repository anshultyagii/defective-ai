import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def tune_model(X, y, groups, cv):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 2, 4)
        }
        model = RandomForestRegressor(**params, random_state=42)
        maes = []
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y.iloc[test_idx], preds))
        return np.mean(maes)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    return study.best_params
