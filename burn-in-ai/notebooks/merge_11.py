import nbformat as nbf
import os
import glob

def merge():
    merged_nb = nbf.v4.new_notebook()
    all_cells = []
    
    # Read the individual notebooks in order
    files = sorted(glob.glob("*.ipynb"))
    
    for f in files:
        nb = nbf.read(f, as_version=4)
        all_cells.extend(nb.cells)
        
    merged_nb['cells'] = all_cells
    
    # Write the merged notebook
    with open('Master_System_Evaluation.ipynb', 'w') as f:
        nbf.write(merged_nb, f)
        
    # Delete the old files
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    merge()
