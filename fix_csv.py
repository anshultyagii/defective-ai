import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv('burn-in-ai/data/processed/mosfet_screening_dataset.csv')
features_anom = ['Value_24h', 'delta_0_24', 'slope_0_24', 'peer_deviation']

iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(df[features_anom])
df['peer_anomaly_score'] = -iso.decision_function(df[features_anom])

df.to_csv('burn-in-ai/data/processed/mosfet_screening_dataset.csv', index=False)
