import os
import glob
import scipy.io
import pandas as pd
import numpy as np

def extract_time_info(data_dir, device_name):
    files = glob.glob(os.path.join(data_dir, f"{device_name}_run_*.mat"))
    files.sort(key=lambda x: int(x.split('_run_')[1].split('.mat')[0]))
    
    for f in files:
        run_num = int(f.split('_run_')[1].split('.mat')[0])
        try:
            mat = scipy.io.loadmat(f)
            meas = mat['measurement'][0,0]
            ss = meas['steadyState']
            t = ss['timeEpoch'][0,0].flatten()
            print(f"Run {run_num}: {len(t)} points, Start {t[0]}, End {t[-1]}, Duration {t[-1] - t[0]} seconds")
        except Exception as e:
            pass

if __name__ == '__main__':
    extract_time_info('../../../../data/raw/mosfet', 'Test_10')
