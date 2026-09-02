import pandas as pd
df = pd.read_csv('burn-in-ai/data/processed/mosfet_screening_dataset.csv')
print(df[['component_id', 'Value_0h', 'Value_168h', 'future_risk_label']].head(10))
print(f"Total rows: {len(df)}")
print(f"Non-NaN 168h: {df['Value_168h'].notna().sum()}")
