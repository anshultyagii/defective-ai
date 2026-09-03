import h5py

with h5py.File(f'/Users/anshul/Documents/AI/data/raw/Capacitor/ES10.mat', 'r') as f:
    c1 = f['ES10']['EIS_Data']['ES10C1']
    print("ES10C1 keys:", list(c1.keys()))
