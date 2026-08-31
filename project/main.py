import pandas as pd
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
    
    feat_df = engineer_features(df)
    X = feat_df[['Value_0h', 'Value_2h', 'Delta_Value', 'Early_Slope']]
    y = feat_df['Target_Value']
    groups = feat_df['Run_ID']
    
    cv = get_cv()
    models = get_models()
    
    res = evaluate_models(models, persistence_baseline, X, y, groups, cv)
    print("\nPHASE 12 - MODEL COMPARISON TABLE:")
    print(res.to_string(index=False))

if __name__ == '__main__':
    main()
