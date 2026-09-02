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
            n_points = ss.shape[1]
            t = [ss[0, i]['timeEpoch'][0,0] for i in range(n_points)]
            print(f"Run {run_num}: {n_points} points, Start {t[0]:.4f}, End {t[-1]:.4f}, Duration {(t[-1] - t[0])*24*3600:.2f} seconds")
        except Exception as e:
            print(e)

if __name__ == '__main__':
    extract_time_info('../../../../data/raw/mosfet', 'Test_10')
