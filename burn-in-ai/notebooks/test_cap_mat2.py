import h5py

with h5py.File('/Users/anshul/Documents/AI/data/raw/Capacitor/ES10.mat', 'r') as f:
    es10 = f['ES10']
    print("ES10 keys:", list(es10.keys()))
