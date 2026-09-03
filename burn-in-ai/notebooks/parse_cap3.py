import h5py
import numpy as np

with h5py.File(f'/Users/anshul/Documents/AI/data/raw/Capacitor/ES10.mat', 'r') as f:
    meas = f['ES10']['EIS_Data']['ES10C1']['EIS_Measurement']
    print("Type:", type(meas))
    if isinstance(meas, h5py.Dataset):
        print("Shape:", meas.shape)
    else:
        print("Keys:", list(meas.keys()))
