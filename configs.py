"""
Central configuration for modal decomposition analysis.
"""
import os, re, time, json, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.linalg import eig
import scipy.linalg
from tqdm import tqdm

os.environ['OS_ACTIVITY_MODE'] = 'disable'  # suppress macOS IMKClient logs
"""
Configuration and shared imports for modal decomposition tools.
"""

# Default directories
RESULTS_DIR = "./preprocess"
FIGURES_DIR = "./figs"
CACHE_DIR = "./cache"

# Figure saving options
FIG_DPI = 300
FIG_FORMAT = "png"  # or "pdf"

# FFT backend: "scipy", "numpy", "tensorflow", "torch", "cv2" (OpenCV)
# Should match the naming used in fft_benchmark.py
FFT_BACKEND = "scipy"  # Default, options: "scipy", "numpy", "tensorflow", "torch", "cv2"

# Matplotlib/LaTeX options
USE_LATEX = False  # Set True to enable LaTeX rendering
FONT_FAMILY = "serif"
FONT_SIZE = 12
CMAP_SEQ = 'viridis'  # Sequential colormap for general use
CMAP_DIV = 'RdBu_r'   # Diverging colormap for signed data

# Default window type for FFT
WINDOW_TYPE = "hamming"
WINDOW_NORM = "power"

# Other global options can be added here as needed
