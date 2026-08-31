import pandas as pd
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
