"""
Compatibility layer for running Portfolio Visualizer in both Jupyter and Google Colab.
This module provides environment detection and compatible widgets/functions.
"""

import os
import sys
from IPython.display import display, HTML, FileLink

# Environment detection
def is_colab():
    """Detect if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Conditional imports based on environment
IS_COLAB = is_colab()

if IS_COLAB:
    from google.colab import widgets as gc_widgets
    from google.colab import files as colab_files
else:
    gc_widgets = None
    colab_files = None

import ipywidgets as widgets
from ipywidgets import Tab, VBox, HBox


class CompatibleTabBar:
    """
    A cross-platform TabBar widget that works in both Jupyter and Colab.
    In Colab: Uses gc_widgets.TabBar
    In Jupyter: Uses ipywidgets.Tab
    """
    
    def __init__(self, tab_names, location='top'):
        """
        Initialize compatible tab bar.
        
        Args:
            tab_names: List of tab names
            location: Tab location (only used in Colab)
        """
        self.tab_names = tab_names
        self.location = location
        self._tab_outputs = []
        
        if IS_COLAB:
            # Use Colab's TabBar
            self._widget = gc_widgets.TabBar(tab_names, location=location)
        else:
            # Use ipywidgets Tab for Jupyter
            # Create output widgets for each tab
            self._tab_outputs = [widgets.Output() for _ in tab_names]
            self._widget = Tab(children=self._tab_outputs)
            for i, name in enumerate(tab_names):
                self._widget.set_title(i, name)
    
    def output_to(self, tab_index):
        """
        Context manager to send output to a specific tab.
        
        Args:
            tab_index: Index of the tab (0-based) or name of the tab
        """
        # Convert string name to index if needed
        if isinstance(tab_index, str):
            try:
                tab_index = self.tab_names.index(tab_index)
            except ValueError:
                raise ValueError(f"Tab '{tab_index}' not found in {self.tab_names}")

        if IS_COLAB:
            return self._widget.output_to(tab_index)
        else:
            # For Jupyter, return the output widget's context
            return self._tab_outputs[tab_index]
    
    def display(self):
        """Display the tab widget"""
        display(self._widget)


def download_file(filename):
    """
    Download a file in a cross-platform way.
    
    Args:
        filename: Path to file to download
    """
    if IS_COLAB:
        # Use Colab's file download
        colab_files.download(filename)
    else:
        # For Jupyter, display a download link
        display(FileLink(filename))
        print(f"Click the link above to download: {filename}")


def get_data_path():
    """
    Get the appropriate data path based on environment.
    
    Returns:
        str: Path to data directory
    """
    if IS_COLAB:
        return '/content/Projects/Data/'
    else:
        # For local Jupyter, assume data is in same directory structure
        # Get the directory of the notebook (current working directory)
        base_path = os.getcwd()
        return os.path.join(base_path, 'Data/')


def get_base_path():
    """
    Get the base project path based on environment.
    
    Returns:
        str: Base project path
    """
    if IS_COLAB:
        return '/content/Projects/'
    else:
        return os.getcwd()


def setup_environment():
    """
    Set up the environment (install packages, clone repo if needed).
    Call this at the start of the notebook.
    """
    if IS_COLAB:
        # Install packages in Colab
        import subprocess
        print("Setting up Colab environment...")
        subprocess.run(["pip", "install", "-q", "reportlab", "kaleido", "PyPDF2"], check=False)
        
        # Clone repository if it doesn't exist
        if not os.path.exists('/content/Projects'):
            subprocess.run(["git", "clone", "https://github.com/gabrielanatalia/Projects/"], check=False)
            os.chdir('/content/Projects')
        
        # Add to path
        if '/content/Projects' not in sys.path:
            sys.path.append('/content/Projects')
    else:
        # For Jupyter, just ensure we're in the right directory
        print("Setting up Jupyter environment...")
        # Check if required packages are installed
        try:
            import reportlab
            import kaleido
            import PyPDF2
        except ImportError as e:
            print(f"Missing package: {e}")
            print("Please install required packages:")
            print("  pip install reportlab kaleido PyPDF2")
        
        # Add current directory to path if not already there
        base_path = get_base_path()
        if base_path not in sys.path:
            sys.path.append(base_path)


def display_styled_dataframe(df, use_heatmap=False):
    """
    Display a styled dataframe that works in both environments.
    
    Args:
        df: DataFrame to display
        use_heatmap: Whether to apply heatmap styling
    """
    if use_heatmap:
        styled = df.style.background_gradient(cmap='Blues', axis=None)
    else:
        styled = df.style
    
    # Both Jupyter and Colab support styled dataframes via display()
    display(styled)


# Print environment info
if __name__ != "__main__":
    env_name = "Google Colab" if IS_COLAB else "Jupyter Notebook"
    print(f"✓ Compatibility layer loaded for: {env_name}")
