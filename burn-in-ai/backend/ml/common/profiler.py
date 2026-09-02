import os
import glob
import scipy.io
import pandas as pd
import numpy as np

def profile_mosfet(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.mat"))
    print("Component: MOSFET")
    print(f"Files found: {len(files)}")
    devices = set([os.path.basename(f).split('_')[1] for f in files if "Test" in f])
    print(f"Devices: {len(devices)}")
    
    # Just inspect the first file for parameters
    mat = scipy.io.loadmat(files[0])
    meas = mat['measurement'][0,0]
    ss = meas['steadyState']
    time_len = ss.shape[1]
    
    params = ss.dtype.names
    try:
        ss_fields = ss[0,0]['timeDomain'].dtype.names
    except:
        ss_fields = ('unknown',)
    
    print(f"Measurements per run: ~{time_len}")
    print(f"Parameters in steadyState timeDomain: {ss_fields}")
    print("Missing values: to be determined")
    print("Failure information: Run to failure (multiple runs per device)")
    print("Time-series: Available\n")

def profile_igbt(data_dir):
    files = []
    for root, _, fs in os.walk(data_dir):
        for f in fs:
            if f.endswith('.mat') or f.endswith('.csv'):
                files.append(os.path.join(root, f))
    print("Component: IGBT")
    print(f"Files found: {len(files)}")
    print("Time-series: Available\n")

def profile_capacitor(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.mat"))
    print("Component: CAPACITOR")
    print(f"Files found: {len(files)}")
    print("Time-series: Available\n")

if __name__ == '__main__':
    profile_mosfet('burn-in-ai/data/raw/mosfet')
    profile_igbt('burn-in-ai/data/raw/igbt')
    profile_capacitor('burn-in-ai/data/raw/capacitor')
