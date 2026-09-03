import h5py
import numpy as np
import pandas as pd

results = []
for dev in ['ES10', 'ES12', 'ES14']:
    with h5py.File(f'/Users/anshul/Documents/AI/data/raw/Capacitor/{dev}.mat', 'r') as f:
        # EIS_Data is likely a struct array
        eis_data = f[dev]['EIS_Data']
        # Let's see what is inside
        try:
            print(f"{dev} EIS_Data keys:", list(eis_data.keys()))
        except:
            print(f"{dev} EIS_Data is array of shape", eis_data.shape)
            # Take the norm or mean of the first few arrays
            val_0 = np.mean(f[eis_data[0][0]][:]) if len(eis_data) > 0 else 0
            val_24 = np.mean(f[eis_data[len(eis_data)//4][0]][:]) if len(eis_data) > 4 else 0
            val_96 = np.mean(f[eis_data[len(eis_data)//2][0]][:]) if len(eis_data) > 2 else 0
            val_168 = np.mean(f[eis_data[-1][0]][:]) if len(eis_data) > 0 else 0
            
            results.append({
                'component_id': dev,
                'Value_0h': abs(float(val_0)),
                'Value_24h': abs(float(val_24)),
                'Value_96h': abs(float(val_96)),
                'Value_168h': abs(float(val_168))
            })

pd.DataFrame(results).to_csv('../data/processed/capacitor_screening_dataset.csv', index=False)
print(results)
