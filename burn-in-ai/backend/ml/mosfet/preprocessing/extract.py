import os
import glob
import scipy.io
import pandas as pd
import numpy as np

def extract_mosfet_device(data_dir, device_name):
    files = glob.glob(os.path.join(data_dir, f"{device_name}_run_*.mat"))
    files.sort(key=lambda x: int(x.split('_run_')[1].split('.mat')[0]))
    
    records = []
    
    for f in files:
        run_num = int(f.split('_run_')[1].split('.mat')[0])
        try:
            mat = scipy.io.loadmat(f)
            meas = mat['measurement'][0,0]
            ss = meas['steadyState']
            
            # steadyState might have multiple fields
            vds = ss['timeDomain'][0,0]['drainSourceVoltage'][0,0].flatten()
            id_ = ss['timeDomain'][0,0]['drainCurrent'][0,0].flatten()
            temp = ss['timeDomain'][0,0]['packageTemperature'][0,0].flatten()
            
            # calculate RDS_on
            rds_on = np.abs(vds / (id_ + 1e-6))
            
            # take mean for the run to simplify
            records.append({
                'device': device_name,
                'run': run_num,
                'vds_mean': np.mean(vds),
                'id_mean': np.mean(id_),
                'temp_mean': np.mean(temp),
                'rds_on_mean': np.mean(rds_on)
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    df = pd.DataFrame(records)
    print(df)
    return df

if __name__ == '__main__':
    extract_mosfet_device('../../../../data/raw/mosfet', 'Test_10')
