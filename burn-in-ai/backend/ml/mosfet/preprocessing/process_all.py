import os
import glob
import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

def process_all_mosfet(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(data_dir, "*.mat"))
    devices = sorted(list(set([os.path.basename(f).split('_run_')[0] for f in files if "Test" in f])))
    
    for device in tqdm(devices):
        device_files = glob.glob(os.path.join(data_dir, f"{device}_run_*.mat"))
        
        valid_files = []
        for f in device_files:
            match = re.search(r'_run_(\d+)\.mat', f)
            if match:
                valid_files.append((f, int(match.group(1))))
        
        valid_files.sort(key=lambda x: x[1])
        
        all_records = []
        start_time = None
        
        for f, run_num in valid_files:
            try:
                mat = scipy.io.loadmat(f)
                if 'measurement' not in mat:
                    continue
                meas = mat['measurement'][0,0]
                if 'steadyState' not in meas.dtype.names:
                    continue
                ss = meas['steadyState']
                n_points = ss.shape[1]
                
                # Sample every 50th point
                for i in range(0, n_points, 50):
                    try:
                        t = ss[0, i]['timeEpoch'][0,0]
                        vds = ss[0, i]['timeDomain'][0,0]['drainSourceVoltage'][0,0].flatten()
                        id_ = ss[0, i]['timeDomain'][0,0]['drainCurrent'][0,0].flatten()
                        temp = ss[0, i]['timeDomain'][0,0]['packageTemperature'][0,0].flatten()
                        
                        vds_mean = np.mean(vds)
                        id_mean = np.mean(id_)
                        temp_mean = np.mean(temp)
                        rds_on = np.abs(vds_mean / (id_mean + 1e-6))
                        
                        if start_time is None:
                            start_time = t
                            
                        time_hours = (t - start_time) * 24
                        
                        all_records.append({
                            'device': device,
                            'run': run_num,
                            'time_hours': time_hours,
                            'vds': vds_mean,
                            'id': id_mean,
                            'temp': temp_mean,
                            'rds_on': rds_on
                        })
                    except Exception as e:
                        pass
            except Exception as e:
                pass
                
        if all_records:
            df = pd.DataFrame(all_records)
            df.to_csv(os.path.join(output_dir, f"{device}.csv"), index=False)

if __name__ == '__main__':
    process_all_mosfet('../../../../data/raw/mosfet', '../../../../data/processed/mosfet')
