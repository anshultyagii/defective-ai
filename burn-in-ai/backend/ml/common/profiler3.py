import os
import glob
import pandas as pd
import numpy as np
import h5py

def profile_capacitor(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.mat"))
    print("Component: CAPACITOR")
    print(f"Files found: {len(files)}")
    if len(files) > 0:
        with h5py.File(files[0], 'r') as f:
            print("Keys in Capacitor file:", list(f.keys()))
            for k in f.keys():
                print(f"  {k}: {type(f[k])}")
    print("Time-series: Available\n")

if __name__ == '__main__':
    profile_capacitor('burn-in-ai/data/raw/capacitor')
