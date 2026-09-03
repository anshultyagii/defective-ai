import h5py
import scipy.io as sio

try:
    mat = sio.loadmat('/Users/anshul/Documents/AI/data/raw/Capacitor/ES10.mat')
    print("Loaded with scipy.io")
except Exception as e:
    print("Scipy failed:", e)
    try:
        with h5py.File('/Users/anshul/Documents/AI/data/raw/Capacitor/ES10.mat', 'r') as f:
            print("Loaded with h5py")
            print("Keys:", list(f.keys()))
    except Exception as e2:
        print("h5py failed:", e2)
