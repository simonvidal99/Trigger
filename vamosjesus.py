import json

# Abre el archivo de Jupyter Notebook
with open('newdbprocessing.ipynb', 'r') as f:
    notebook = json.load(f)

# Recorre todas las celdas
for cell in notebook['cells']:
    # Si la celda es de tipo 'code' y no tiene el campo 'outputs', añádelo
    if cell['cell_type'] == 'code' and 'outputs' not in cell:
        cell['outputs'] = []

# Guarda el archivo de Jupyter Notebook
with open('your_notebook.ipynb', 'w') as f:
    json.dump(notebook, f)
