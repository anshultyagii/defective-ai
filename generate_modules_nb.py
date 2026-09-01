import nbformat as nbf

# ==========================================
# MODULE A: Outlier Detection System
# ==========================================
nb_a = nbf.v4.new_notebook()
nb_a.cells = [
    nbf.v4.new_code_cell('!pip install pandas numpy scikit-learn matplotlib seaborn'),
    nbf.v4.new_markdown_cell('# MODULE A: Dynamic Population-Based Outlier Detection\nThis module detects components whose measurements are abnormal relative to the healthy population (rather than relying on static datasheet limits). It uses an **Isolation Forest** machine learning model to find multivariate anomalies in the early-life data.'),
    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

# 1. Extract Early-Life Features for Population Comparison
file_paths = sorted(glob.glob('/Users/anshul/Documents/AI/data/raw/MOSFET/MOSFET_Device_*.csv'))
population_data = []

for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['Aging_Hours', 'RDS_on_Ohm', 'Package_Temp_C']).sort_values('Aging_Hours')
        if len(df) < 5: continue
            
        # Look only at the first few hours to emulate static screening timeframes (e.g., first 24h)
        early_df = df[df['Aging_Hours'] <= 24.0]
        if len(early_df) < 2: 
            early_df = df.head(10) # Fallback if component died very early
            
        population_data.append({
            'Device_ID': str(df['Device_ID'].iloc[0]),
            'Initial_RDS_Ohm': early_df['RDS_on_Ohm'].iloc[0],
            'Early_Max_RDS': early_df['RDS_on_Ohm'].max(),
            'Early_Temp_Variance': early_df['Package_Temp_C'].var(),
            'Early_RDS_Variance': early_df['RDS_on_Ohm'].var()
        })
    except Exception as e:
        pass

pop_df = pd.DataFrame(population_data).fillna(0)

# 2. Train Isolation Forest (Unsupervised Anomaly Detection)
# We assume approx 10% of components might be latent outliers
iso_forest = IsolationForest(contamination=0.10, random_state=42)

features = pop_df[['Initial_RDS_Ohm', 'Early_Max_RDS', 'Early_Temp_Variance', 'Early_RDS_Variance']]
pop_df['Anomaly_Score'] = iso_forest.fit_predict(features)
# Anomaly_Score: -1 means OUTLIER, 1 means INLIER

pop_df['Decision'] = pop_df['Anomaly_Score'].apply(lambda x: 'FLAG (Outlier)' if x == -1 else 'PASS (Normal)')

print("--- MODULE A: OUTLIER SCREENING RESULTS ---")
print(pop_df[['Device_ID', 'Decision', 'Initial_RDS_Ohm', 'Early_Temp_Variance']].to_string(index=False))

# 3. Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pop_df, x='Initial_RDS_Ohm', y='Early_Temp_Variance', hue='Decision', palette={'PASS (Normal)': 'green', 'FLAG (Outlier)': 'red'}, s=100)
plt.title('Module A: Population-Based Outlier Detection')
plt.xlabel('Initial Resistance (Ohms)')
plt.ylabel('Early Temperature Variance')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
''')
]
with open('/Users/anshul/Documents/AI/notebooks/MODULE_A_Outlier_Detection.ipynb', 'w') as f:
    nbf.write(nb_a, f)


# ==========================================
# MODULE B: Drift Predictor & Explainer
# ==========================================
nb_b = nbf.v4.new_notebook()
nb_b.cells = [
    nbf.v4.new_code_cell('!pip install pandas numpy scikit-learn shap matplotlib'),
    nbf.v4.new_markdown_cell('# MODULE B: Time-Series Drift Predictor & Explainer\nThis module uses early-life data to predict if a component will suffer catastrophic drift in the future. It then uses **SHAP (SHapley Additive exPlanations)** to output a human-readable engineering explanation for *every single component individually*.'),
    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings; warnings.filterwarnings('ignore')

# 1. Build the Global ML Dataset
file_paths = sorted(glob.glob('/Users/anshul/Documents/AI/data/raw/MOSFET/MOSFET_Device_*.csv'))
dataset = []

for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['Aging_Hours', 'RDS_on_Ohm', 'Package_Temp_C']).sort_values('Aging_Hours')
        if len(df) < 10: continue
            
        cutoff = df['Aging_Hours'].quantile(0.8)
        train = df[df['Aging_Hours'] <= cutoff]
        test = df[df['Aging_Hours'] > cutoff]
        
        if len(test) == 0 or len(train) < 5: continue
            
        x_hours = train['Aging_Hours'].values
        y_rds = train['RDS_on_Ohm'].values
        slope = np.polyfit(x_hours, y_rds, 1)[0] if len(x_hours) > 1 else 0
        
        target = int(test['RDS_on_Ohm'].max() > train['RDS_on_Ohm'].max() * 1.05)
        
        dataset.append({
            'Device_ID': str(df['Device_ID'].iloc[0]),
            'Early_Slope_RDS': slope,
            'Early_Var_RDS': train['RDS_on_Ohm'].var(),
            'Early_Max_RDS': train['RDS_on_Ohm'].max(),
            'Early_Max_Temp': train['Package_Temp_C'].max(),
            'Early_Var_Temp': train['Package_Temp_C'].var(),
            'Target_Failure': target
        })
    except Exception as e:
        pass

ml_df = pd.DataFrame(dataset).fillna(0)
X = ml_df.drop(columns=['Device_ID', 'Target_Failure'])
y = ml_df['Target_Failure']

# 2. Train the Drift Predictor (Random Forest)
clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
clf.fit(X, y)

ml_df['Prediction'] = clf.predict(X)
ml_df['Probability'] = clf.predict_proba(X)[:, 1]

# 3. Generate SHAP Explanations
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

# Handle the structure of shap_values (different sklearn versions return different structures for classification)
if isinstance(shap_values, list):
    # For binary classification, we want the SHAP values for class 1 (Failure)
    shap_vals_class1 = shap_values[1]
else:
    # Some newer versions might return a single array for binary classification, or a 3D array
    if len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values

print("======================================================")
print(" MODULE B: DRIFT PREDICTION & COMPONENT EXPLANATIONS  ")
print("======================================================")

for i in range(len(ml_df)):
    device = ml_df['Device_ID'].iloc[i]
    pred = "FLAG (High Risk of Drift)" if ml_df['Prediction'].iloc[i] == 1 else "PASS (Stable)"
    prob = ml_df['Probability'].iloc[i] * 100
    
    print(f"\\n--- COMPONENT: Device {device} ---")
    print(f"Prediction: {pred} (Failure Probability: {prob:.1f}%)")
    
    # Extract feature contributions for this specific component
    component_shap = shap_vals_class1[i]
    feature_names = X.columns
    
    # Sort features by absolute contribution magnitude
    contributions = list(zip(feature_names, component_shap, X.iloc[i].values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Top 3 Driving Factors for this Decision:")
    for rank in range(min(3, len(contributions))):
        feat_name = contributions[rank][0]
        shap_val = contributions[rank][1]
        feat_val = contributions[rank][2]
        
        effect = "Increased Risk" if shap_val > 0 else "Decreased Risk (Stabilizing)"
        
        print(f"  {rank+1}. {feat_name}: {feat_val:.5f} -> {effect} (Impact: {shap_val:+.4f})")
    
    print("-" * 54)
''')
]
with open('/Users/anshul/Documents/AI/notebooks/MODULE_A_Outlier_Detection.ipynb', 'w') as f:
    nbf.write(nb_a, f)
with open('/Users/anshul/Documents/AI/notebooks/MODULE_B_Drift_Predictor.ipynb', 'w') as f:
    nbf.write(nb_b, f)

print("Modules generated successfully.")
