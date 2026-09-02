import nbformat as nbf
import os
import glob

def merge():
    merged_nb = nbf.v4.new_notebook()
    all_cells = []
    
    # Read the individual notebooks in order
    files = sorted(glob.glob("0*_MOSFET_*.ipynb"))
    
    for f in files:
        nb = nbf.read(f, as_version=4)
        # Add a markdown cell to act as a separator if we want, but the notebooks already have headers
        all_cells.extend(nb.cells)
        
    merged_nb['cells'] = all_cells
    
    # Write the merged notebook
    with open('MOSFET_End_to_End_Analysis.ipynb', 'w') as f:
        nbf.write(merged_nb, f)
        
    # Delete the old files
    for f in files:
        os.remove(f)
        
    # Delete the old generic notebooks if they exist
    for f in ["MOSFET_Anomaly_Evaluation.ipynb", "MOSFET_Drift_Evaluation.ipynb"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == '__main__':
    merge()
    print("Successfully merged into MOSFET_End_to_End_Analysis.ipynb and deleted fragments.")
