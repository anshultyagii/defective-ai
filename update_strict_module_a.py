import nbformat as nbf

nb_a_path = '/Users/anshul/Documents/AI/notebooks/MODULE_A_Outlier_Detection.ipynb'
with open(nb_a_path, 'r') as f:
    nb_a = nbf.read(f, as_version=4)

module_a_explanation = nbf.v4.new_markdown_cell('''### 🟢 The Terms of PASS / FLAG (How the Logic Works)
**Problem Statement Alignment:** "Participants need to develop a 'Dynamic' outlier detection system. If a lot has an average leakage current of 10µA, a part showing 45 µA is a massive anomaly, even if the absolute datasheet maximum limit is 50 µA."

This module does exactly that: it ignores static absolute limits and evaluates if a component is behaving abnormally compared to the current population average.
* **The Algorithm:** It uses an unsupervised machine learning model called **Isolation Forest**.
* **The Terms:** The model evaluates early-life features to measure how mathematically 'isolated' a component is from the main herd's average behavior.
* **FLAG (Outlier):** If a component's measurements are so unusual that they land in the top 10% most isolated data points in the population (`contamination=0.10`), the algorithm assigns it an anomaly score of `-1` and FLAGS it (just like the 45µA example in a 10µA average lot).
* **PASS (Normal):** If the component's measurements sit tightly inside the dense cluster of normal average behavior exhibited by the other components, it receives an anomaly score of `1` and PASSES.
''')

# Replace the previous explanation cell
nb_a.cells[2] = module_a_explanation

with open(nb_a_path, 'w') as f:
    nbf.write(nb_a, f)
print("Module A updated.")
