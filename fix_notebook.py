import nbformat as nbf
import os

with open('burn-in-ai/notebooks/MOSFET_End_to_End_Analysis.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# We want to remove the repetitive `df = pd.read_csv(...)` in every cell after the first one.
# OR just ensure `peer_anomaly_score` gets added to df and saved.
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "df['peer_anomaly_score'] = -iso.decision_function(df[features_anom])" in cell.source:
            cell.source = cell.source.replace(
                "df['peer_anomaly_score'] = -iso.decision_function(df[features_anom])\nprint('Anomaly Models Trained')",
                "df['peer_anomaly_score'] = -iso.decision_function(df[features_anom])\ndf.to_csv('../data/processed/mosfet_screening_dataset.csv', index=False)\nprint('Anomaly Models Trained')"
            )

with open('burn-in-ai/notebooks/MOSFET_End_to_End_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
