from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def get_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(random_state=42)
    except ImportError as e:
        print("XGBoost not available, skipping.")
    except Exception as e:
        print(f"Failed to load XGBoost: {e}")
    return models
