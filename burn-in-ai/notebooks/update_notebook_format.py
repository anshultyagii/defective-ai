import nbformat as nbf

with open('Master_Orchestration.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and 'recall_score' in cell.source:
        source = cell.source
        
        # Replace the formatting strings to display percentages
        source = source.replace("print(f\"Recall: {recall_score(y_true, static_preds, zero_division=0):.4f}\")", 
                                "print(f\"Recall: {recall_score(y_true, static_preds, zero_division=0) * 100:.2f}%\")")
        source = source.replace("print(f\"Accuracy: {accuracy_score(y_true, static_preds):.4f}\")", 
                                "print(f\"Accuracy: {accuracy_score(y_true, static_preds) * 100:.2f}%\")")
                                
        source = source.replace("print(f\"Recall: {recall_score(y_true, preds, zero_division=0):.4f}\")", 
                                "print(f\"Recall: {recall_score(y_true, preds, zero_division=0) * 100:.2f}%\")")
        source = source.replace("print(f\"Accuracy: {accuracy_score(y_true, preds):.4f}\")", 
                                "print(f\"Accuracy: {accuracy_score(y_true, preds) * 100:.2f}%\")")
        source = source.replace("print(f\"Precision: {precision_score(y_true, preds, zero_division=0):.4f}\")", 
                                "print(f\"Precision: {precision_score(y_true, preds, zero_division=0) * 100:.2f}%\")")
        
        cell.source = source

with open('Master_Orchestration.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Updated formatting.")
