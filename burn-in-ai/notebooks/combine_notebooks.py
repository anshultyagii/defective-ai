import nbformat as nbf
import glob
import os

def merge_notebooks():
    merged_nb = nbf.v4.new_notebook()
    all_cells = []
    
    # Sort files to maintain 01 to 11 order
    files = sorted(glob.glob("*.ipynb"))
    
    for f in files:
        if f == "Master_Orchestration.ipynb": continue
        nb = nbf.read(f, as_version=4)
        all_cells.extend(nb.cells)
        
    merged_nb['cells'] = all_cells
    
    with open('Master_Orchestration.ipynb', 'w') as f:
        nbf.write(merged_nb, f)
        
    print(f"Merged {len(files)} notebooks into Master_Orchestration.ipynb")

if __name__ == '__main__':
    merge_notebooks()
