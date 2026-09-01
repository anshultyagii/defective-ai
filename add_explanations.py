import nbformat as nbf

# ----- Update Module A -----
nb_a_path = '/Users/anshul/Documents/AI/notebooks/MODULE_A_Outlier_Detection.ipynb'
with open(nb_a_path, 'r') as f:
    nb_a = nbf.read(f, as_version=4)

module_a_explanation = nbf.v4.new_markdown_cell('''### 🟢 The Terms of PASS / FLAG (How the Logic Works)
This module does not predict the future; it evaluates if a component is behaving abnormally compared to its peers right now. 
* **The Algorithm:** It uses an unsupervised machine learning model called **Isolation Forest**.
* **The Terms:** The model evaluates 4 early-life features (Initial Resistance, Max Early Resistance, Temperature Variance, and Resistance Variance) to measure how mathematically 'isolated' a component is from the main herd.
* **FLAG (Outlier):** If a component's measurements are so unusual that they land in the top 10% most isolated data points in the population (`contamination=0.10`), the algorithm assigns it an anomaly score of `-1` and FLAGS it.
* **PASS (Normal):** If the component's measurements sit tightly inside the dense cluster of normal behavior exhibited by the other components, it receives an anomaly score of `1` and PASSES.
''')

# Avoid inserting twice if the user runs it twice, check if it already exists
if "🟢 The Terms of PASS" not in nb_a.cells[2].source:
    nb_a.cells.insert(2, module_a_explanation)
else:
    nb_a.cells[2] = module_a_explanation

with open(nb_a_path, 'w') as f:
    nbf.write(nb_a, f)


# ----- Update Module B -----
nb_b_path = '/Users/anshul/Documents/AI/notebooks/MODULE_B_Drift_Predictor.ipynb'
with open(nb_b_path, 'r') as f:
    nb_b = nbf.read(f, as_version=4)

module_b_explanation = nbf.v4.new_markdown_cell('''### 🔵 The Terms of PASS / FLAG (How the Logic Works)
This module actively looks into the future to predict catastrophic failure.
* **The Target Definition (What is a failure?):** A failure is mathematically defined as a component whose resistance suddenly spikes more than **5% above its own early-life maximum** during the final 20% of its testing time.
* **The Algorithm:** It uses a **Random Forest Classifier** trained to find hidden early-life micro-patterns (like temperature deltas and variance) that cause that 5% spike later on.
* **FLAG (High Risk of Drift):** The model predicts a `1`. This means the AI is highly confident that this specific component **WILL** suffer that >5% exponential degradation spike later in its lifespan, even if it looks perfectly healthy right now.
* **PASS (Stable):** The model predicts a `0`. This means the component's early-life variance matches the pattern of healthy components that safely survived their entire lifespan without spiking.
''')

if "🔵 The Terms of PASS" not in nb_b.cells[2].source:
    nb_b.cells.insert(2, module_b_explanation)
else:
    nb_b.cells[2] = module_b_explanation

with open(nb_b_path, 'w') as f:
    nbf.write(nb_b, f)

print('Successfully updated notebooks without bash errors.')
