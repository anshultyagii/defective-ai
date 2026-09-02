import os
import glob
import pandas as pd
import numpy as np

def build_mosfet_features(input_dir, output_file):
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    all_data = []
    
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        
        # Sort by time
        df = df.sort_values('time_hours').reset_index(drop=True)
        
        # Initial value of RDS_on (mean of first few points to reduce noise)
        initial_rds = df['rds_on'].iloc[:5].mean()
        
        # Normalized RDS
        df['rds_on_norm'] = df['rds_on'] / initial_rds
        
        # Calculate drift rate (derivative)
        # Using a rolling window for smoother derivative
        df['rds_drift_rate'] = df['rds_on'].diff() / df['time_hours'].diff()
        df['rds_drift_rate'] = df['rds_drift_rate'].rolling(window=5, min_periods=1).mean().fillna(0)
        
        # Calculate drift acceleration (second derivative)
        df['rds_drift_accel'] = df['rds_drift_rate'].diff() / df['time_hours'].diff()
        df['rds_drift_accel'] = df['rds_drift_accel'].rolling(window=5, min_periods=1).mean().fillna(0)
        
        # Time-based differences (ΔRDS at certain hours) - we can just look back in time
        # Or we can interpolate for specific checkpoints
        # Let's add rolling percentage change
        
        all_data.append(df)
        
    if not all_data:
        print("No data to process")
        return
        
    combined = pd.concat(all_data, ignore_index=True)
    
    # Calculate Lot-relative RDS deviation
    # Group by a time bucket (e.g., every 1 hour) to find lot median
    combined['time_bucket'] = combined['time_hours'].round(0)
    lot_stats = combined.groupby('time_bucket')['rds_on'].agg(['median', 'std']).reset_index()
    lot_stats.rename(columns={'median': 'lot_median_rds', 'std': 'lot_std_rds'}, inplace=True)
    
    combined = pd.merge(combined, lot_stats, on='time_bucket', how='left')
    combined['peer_deviation'] = np.abs(combined['rds_on'] - combined['lot_median_rds']) / (combined['lot_std_rds'] + 1e-6)
    
    # Save features
    combined.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    
if __name__ == '__main__':
    build_mosfet_features('../../../../data/processed/mosfet', '../../../../data/processed/mosfet_features.csv')
