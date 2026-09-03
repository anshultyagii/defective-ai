import h5py
import numpy as np

with h5py.File(f'/Users/anshul/Documents/AI/data/raw/Capacitor/ES10.mat', 'r') as f:
    data = f['ES10']['EIS_Data']['ES10C1']['EIS_Measurement']['Data']
    print("Shape:", data.shape)
    # The array probably contains object references to matrices
    print("Content 0:", f[data[0][0]].shape if len(data) > 0 else "Empty")
