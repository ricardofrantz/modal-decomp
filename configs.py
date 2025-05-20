"""
Central configuration for modal decomposition analysis.
"""
import os, re, time
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.linalg import eig

os.environ['OS_ACTIVITY_MODE'] = 'disable'  # suppress macOS IMKClient logs

# Default directories
RESULTS_DIR = "./preprocess"
FIGURES_DIR = "./figs"

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

# Default window type for FFT
WINDOW_TYPE = "hamming"
WINDOW_NORM = "power"

# Other global options can be added here as needed
