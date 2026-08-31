def validate_schema(df):
    assert 'Device_ID' in df.columns
    assert 'Run_ID' in df.columns
    assert 'Aging_Hours' in df.columns
    assert 'RDS_on_Ohm' in df.columns
    print("Schema validated.")
