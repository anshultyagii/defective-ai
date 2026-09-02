with open('burn-in-ai/backend/ml/mosfet/preprocessing/rebuild_pipeline.py', 'r') as f:
    code = f.read()

old_code = """max_real_time = raw_df['time_hours'].max()
time_scale_factor = 200.0 / max_real_time"""

new_code = """device_lifetimes = raw_df.groupby('device')['time_hours'].max()
p20_lifetime = device_lifetimes.quantile(0.2)
# Scale so that 80% of devices survive past 200 EBH
time_scale_factor = 200.0 / p20_lifetime"""

code = code.replace(old_code, new_code)

with open('burn-in-ai/backend/ml/mosfet/preprocessing/rebuild_pipeline.py', 'w') as f:
    f.write(code)
