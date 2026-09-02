import os
import glob
import scipy.io
import pandas as pd
import numpy as np

def profile_igbt(data_dir):
    files = []
    for root, _, fs in os.walk(data_dir, followlinks=True):
        for f in fs:
            if f.endswith('.mat') or f.endswith('.csv'):
                files.append(os.path.join(root, f))
    print("Component: IGBT")
    print(f"Files found: {len(files)}")
    if len(files) > 0:
        print(f"Sample files: {files[:3]}")
    print("Time-series: Available\n")

def profile_capacitor(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.mat"))
    print("Component: CAPACITOR")
    print(f"Files found: {len(files)}")
    if len(files) > 0:
        mat = scipy.io.loadmat(files[0])
        print("Keys:", [k for k in mat.keys() if not k.startswith('__')])
    print("Time-series: Available\n")

if __name__ == '__main__':
    profile_igbt('burn-in-ai/data/raw/igbt')
    profile_capacitor('burn-in-ai/data/raw/capacitor')
