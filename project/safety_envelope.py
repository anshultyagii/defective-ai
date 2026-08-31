def apply_safety_envelope(df, predicted_col, boundary):
    df['Flag'] = (df[predicted_col] > boundary).astype(int)
    return df
