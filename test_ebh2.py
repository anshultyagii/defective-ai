import pandas as pd
df = pd.read_csv('burn-in-ai/data/processed/mosfet_screening_dataset.csv')
print(f"Total rows: {len(df)}")
print(f"Non-NaN 0h: {df['Value_0h'].notna().sum()}")
print(f"Non-NaN 24h: {df['Value_24h'].notna().sum()}")
print(f"Non-NaN 96h: {df['Value_96h'].notna().sum()}")
print(f"Non-NaN 168h: {df['Value_168h'].notna().sum()}")
