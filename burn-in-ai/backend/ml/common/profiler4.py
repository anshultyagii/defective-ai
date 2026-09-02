import os
import glob
import h5py

def profile_capacitor(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.mat"))
    if len(files) > 0:
        with h5py.File(files[0], 'r') as f:
            group_name = [k for k in f.keys() if k != '#refs#'][0]
            print(f"Keys in {group_name}:", list(f[group_name].keys()))
            for k in f[group_name].keys():
                try:
                    if isinstance(f[group_name][k], h5py.Group):
                        print(f"  {k} is Group, subkeys: {list(f[group_name][k].keys())[:5]}")
                    elif isinstance(f[group_name][k], h5py.Dataset):
                        print(f"  {k} is Dataset, shape: {f[group_name][k].shape}")
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    profile_capacitor('burn-in-ai/data/raw/capacitor')
