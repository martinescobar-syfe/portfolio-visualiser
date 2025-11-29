"""
Script to patch the Portfolio Visualizer notebook for Jupyter/Colab compatibility.
This script modifies the notebook in-place to use the compatibility layer.
"""

import json
import re

def patch_notebook(input_notebook, output_notebook):
    """
    Patch a Colab notebook to work in both Jupyter and Colab.
    
    Args:
        input_notebook: Path to original notebook
        output_notebook: Path to save patched notebook
    """
    
    print(f"Loading notebook: {input_notebook}")
    with open(input_notebook, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the first code cell (setup cell)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check if this is the setup cell
            if 'Load libraries and functions' in source or 'pip install' in source:
                print("Found setup cell, patching...")
                
                # Replace Colab-specific imports
                new_source = []
                skip_next = False
                
                for i, line in enumerate(cell['source']):
                    # Skip git clone and cd commands - we'll handle them in compatibility layer
                    if '!git clone' in line or '%cd /content/Projects' in line:
                        continue
                    
                    # Replace colab-specific imports with compatibility layer
                    if 'from google.colab import widgets as gc_widgets' in line:
                        # Add compatibility layer import instead
                        if i == 0 or 'import jupyter_colab_compat' not in ''.join(new_source):
                            new_source.append('# Compatibility layer for Jupyter and Colab\n')
                            new_source.append('import jupyter_colab_compat as jcc\n')
                            new_source.append('jcc.setup_environment()\n')
                            new_source.append('\n')
                        continue
                    
                    if 'from google.colab import files' in line:
                        continue  # Skip, handled by compatibility layer
                    
                    # Update data path
                    if "data_path = '/content/Projects/Data/'" in line:
                        new_source.append('data_path = jcc.get_data_path()\n')
                        continue
                    
                    # Update sys.path.append
                    if "sys.path.append('/content/Projects')" in line:
                        new_source.append('sys.path.append(jcc.get_base_path())\n')
                        continue
                    
                    # Keep other lines
                    new_source.append(line)
                
                cell['source'] = new_source
                print("✓ Setup cell patched")
                break
    
    # Find and patch all gc_widgets.TabBar usages
    patches_made = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell['source'])
            
            if 'gc_widgets.TabBar' in source_text:
                print(f"Found TabBar usage in cell, patching...")
                new_source = []
                
                for line in cell['source']:
                    # Replace gc_widgets.TabBar with jcc.CompatibleTabBar
                    if 'gc_widgets.TabBar' in line:
                        patched_line = line.replace('gc_widgets.TabBar', 'jcc.CompatibleTabBar')
                        new_source.append(patched_line)
                        patches_made += 1
                    # Replace .output_to for compatible usage  
                    elif '.output_to(' in line and 't.' in line:
                        # Keep as is - compatibility layer handles this
                        new_source.append(line)
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
    
    print(f"✓ Patched {patches_made} TabBar usages")
    
    # Save patched notebook
    print(f"Saving patched notebook to: {output_notebook}")
    with open(output_notebook, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print("✓ Notebook patching complete!")
    print("\nNext steps:")
    print("1. Make sure jupyter_colab_compat.py is in the same directory as the notebook")
    print("2. For Jupyter: Ensure Data/ folder and port_cons.py are in the same directory")
    print("3. Open the patched notebook and run all cells")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_nb = sys.argv[1]
        output_nb = sys.argv[2] if len(sys.argv) > 2 else input_nb.replace('.ipynb', '_patched.ipynb')
    else:
        input_nb = 'Portfolio_Visualizer_colab.ipynb'
        output_nb = 'Portfolio_Visualizer.ipynb'  # New name without "_colab"
    
    patch_notebook(input_nb, output_nb)
