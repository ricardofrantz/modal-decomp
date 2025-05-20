#!/usr/bin/env python3
"""
Simple Spectral Proper Orthogonal Decomposition (SPOD) Analysis Script

Author: R. Frantz

Reference codes:
    - https://github.com/SpectralPOD/spod_matlab/tree/master
    - https://github.com/MathEXLab/PySPOD/blob/main/tutorials/tutorial1/tutorial1.ipynb
"""

import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.signal import find_peaks
import re

def load_jetles_data(file_path):
    """Load and preprocess data from HDF5 file."""
    print(f"Loading data from {file_path}")
    with h5py.File(file_path, "r") as fread:
        # Get dimensions and data
        q = fread["p"][:]  # pressure field (or other variable)
        x = fread["x"][:, 0]  # x-coordinates (axial)
        y = fread["r"][0, :]  # y-coordinates (radial)
        dt = fread["dt"][0][0]  # time step
        
    # Transpose q: original shape (Nx, Ny, Ns) -> (Ns, Nx, Ny)
    # Reshape q: flatten spatial dimensions -> (Ns, Nx * Ny) for SPOD
    q = np.transpose(q, (2, 0, 1))
    Nx, Ny = x.shape[0], y.shape[0]
    Ns = q.shape[0]  # number of snapshots (time steps)
    q_reshaped = q.reshape(Ns, Nx * Ny)  # Use reshape for safety
    
    # Return all the data in a dictionary for easy access
    return {
        'q': q_reshaped,  # Data reshaped for SPOD: [time, space]
        'x': x,           # Axial coordinates
        'y': y,           # Radial coordinates
        'dt': dt,         # Time step
        'Nx': Nx,         # Number of points in x
        'Ny': Ny,         # Number of points in y (radial)
        'Ns': Ns          # Number of snapshots
    }

def sine_window(n):
    """
    Return a sine window of length n.
    
    Mathematical form:
        w[j] = sin(pi * (j + 0.5) / n), for j = 0, ..., n-1
    
    This window is used in spectral estimation to reduce spectral leakage and
    is particularly effective for data with sharp tonal peaks (as in cavity flows).
    The sine window has minimal side lobes in the frequency domain, providing
    sharper frequency resolution for tonal features compared to Hamming or Hann windows.
    
    In the context of SPOD for cavity flows, the sine window is used to match
    the reference MATLAB implementation and published studies, where it has been
    shown to improve the detection of tonal modes (e.g., Rossiter modes).
    For jet cases or broadband flows, the Hamming window is typically preferred
    for its smoother spectral averaging properties.
    """
    return np.sin(np.pi * (np.arange(n) + 0.5) / n)

def load_mat_data(file_path):
    """Flexible loader for .mat files with different variable names and shapes."""
    with h5py.File(file_path, "r") as fread:
        # Try common variable names for the main field
        for var in ["p", "u", "v", "data"]:
            if var in fread:
                q = fread[var][:]
                break
        else:
            raise KeyError("No recognized data variable ('p', 'u', 'v', 'data') in file.")
        x = fread["x"][:]
        y = fread["y"][:]
        dt = np.array(fread["dt"])[0][0] if "dt" in fread else 1.0
    print(f"Loaded variable shape: q={q.shape}, x={x.shape}, y={y.shape}")
    # If x and y are 2D (meshgrid), reduce to 1D vectors
    if x.ndim == 2:
        x_vec = x[:,0]
    else:
        x_vec = x
    if y.ndim == 2:
        y_vec = y[0,:]
    else:
        y_vec = y
    Nx, Ny = x_vec.shape[0], y_vec.shape[0]
    # Special handling for (Nx, Ny, Ns)
    if q.shape == (Nx, Ny, q.shape[2]):
        Ns = q.shape[2]
        q = np.transpose(q, (2, 0, 1))  # (Ns, Nx, Ny)
        q_reshaped = q.reshape(Ns, Nx*Ny)
        print(f"Data interpreted as (Nx, Ny, Ns) and transposed to (Ns, Nx, Ny) = {q.shape}")
    # Standard (Ns, Nx, Ny)
    elif q.shape == (q.shape[0], Nx, Ny):
        Ns = q.shape[0]
        q_reshaped = q.reshape(Ns, Nx*Ny)
        print(f"Data interpreted as (Ns, Nx, Ny) = {q.shape}")
    # Try all permutations if above does not match
    else:
        for axes in [(0,1,2),(2,0,1),(2,1,0),(0,2,1),(1,0,2),(1,2,0)]:
            try:
                arr = np.transpose(q, axes)
                Ns, Nxx, Nyy = arr.shape
                if Nxx == Nx and Nyy == Ny:
                    q_reshaped = arr.reshape(Ns, Nx*Ny)
                    print(f"Data interpreted as (Ns, Nx, Ny) = {arr.shape} via permutation {axes}")
                    break
            except Exception:
                continue
        else:
            # Try if already 2D (Ns, Nspace)
            if q.ndim == 2 and q.shape[1] == Nx*Ny:
                q_reshaped = q
                Ns = q.shape[0]
                print(f"Data interpreted as (Ns, Nspace) = {q.shape}")
            else:
                raise ValueError(f"Cannot interpret data shape: q={q.shape}, x={x.shape}, y={y.shape}. Please check the file.")
    return {
        'q': q_reshaped,
        'x': x_vec,
        'y': y_vec,
        'dt': dt,
        'Nx': Nx,
        'Ny': Ny,
        'Ns': q_reshaped.shape[0]
    }

def calculate_polar_weights(x, y):
    """Calculate integration weights for a 2D cylindrical grid (x, r).
    
    These weights represent the approximate area element (r dr dx) 
    associated with each grid point, needed for spatial inner products.
    """
    Nx, Ny = x.shape[0], y.shape[0]
    
    # Calculate y-direction (r-direction) integration weights (Wy)
    # Approximates the area of annular rings: pi * (r_outer^2 - r_inner^2)
    Wy = np.zeros((Ny, 1))
    
    # First point (centerline): area of the first cylindrical segment
    if Ny > 1:
        y_mid_right = (y[0] + y[1]) / 2
        Wy[0] = np.pi * y_mid_right**2
    else:
        Wy[0] = np.pi * y[0]**2
    
    # Middle points: area of annular rings between midpoints
    for i in range(1, Ny - 1):
        y_mid_left = (y[i-1] + y[i]) / 2 # Inner midpoint radius
        y_mid_right = (y[i] + y[i+1]) / 2 # Outer midpoint radius
        Wy[i] = np.pi * (y_mid_right**2 - y_mid_left**2)
    
    # Last point: area of the outermost annular ring segment
    if Ny > 1:
        y_mid_left = (y[-2] + y[-1]) / 2 # Inner midpoint radius
        # Outer boundary is y[-1]
        Wy[Ny - 1] = np.pi * (y[-1]**2 - y_mid_left**2) 
        # Note: A different boundary condition might be needed depending on grid setup.
        # This assumes y[-1] is the outer edge of the last cell.

    # Calculate x-direction integration weights (Wx)
    # Uses trapezoidal/midpoint rule: dx contribution for each x point
    Wx = np.zeros((Nx, 1))
    
    # First point
    if Nx > 1:
        Wx[0] = (x[1] - x[0]) / 2
    else:
        Wx[0] = 1.0
    
    # Middle points
    for i in range(1, Nx - 1):
        Wx[i] = (x[i+1] - x[i-1]) / 2 # Average of adjacent interval widths
    
    # Last point
    if Nx > 1:
        Wx[Nx - 1] = (x[Nx - 1] - x[Nx - 2]) / 2 # Half of last interval dx
    
    # Combine weights: W[i,j] = Wx[i] * Wy[j]
    # Reshape to a column vector (Nx*Ny, 1) to match flattened data q_reshaped
    W = np.reshape(Wx @ np.transpose(Wy), (Nx * Ny, 1))
    
    return W

def calculate_uniform_weights(x, y):
    """Return uniform weights for a 2D grid (Cartesian)."""
    Nx, Ny = x.shape[0], y.shape[0]
    return np.ones((Nx * Ny, 1))

def blocksfft(q, nfft, nblocks, novlap, blockwise_mean=False, normvar=False, window_norm='power', window_type='hamming'):
    """Compute blocked FFT using Welch's method for CSD estimation.

    Args:
        q (np.ndarray): Input data matrix [time, space].
        nfft (int): Number of snapshots per block (FFT length).
        nblocks (int): Total number of blocks.
        novlap (int): Number of overlapping snapshots between blocks.
        blockwise_mean (bool): If True, subtract blockwise mean instead of global mean.
        normvar (bool): If True, normalize each block by pointwise variance.
        window_norm (str): 'power' (default) or 'amplitude'.
        window_type (str): 'hamming' (default) or 'sine'.

    Returns:
        np.ndarray: FFT coefficients array [frequency, space, block].
    """
    # Select window function
    if window_type == 'sine':
        window = sine_window(nfft)
    else:
        window = np.hamming(nfft)

    if window_norm == 'amplitude':
        cw = 1.0 / window.mean()
    else:  # 'power' normalization (default)
        cw = 1.0 / np.sqrt(np.mean(window**2))

    nmesh = q.shape[1] # Number of spatial points (Nx * Ny)
    q_hat = np.zeros((nfft, nmesh, nblocks), dtype=complex)
    q_mean = np.mean(q, axis=0) # Temporal mean (long-time mean)
    window_broadcast = window[:, np.newaxis] # Reshape window for broadcasting

    for iblk in range(nblocks):
        ts = min(iblk * (nfft - novlap), q.shape[0] - nfft) # Start index
        tf = np.arange(ts, ts + nfft) # Time indices for the block
        block = q[tf, :]
        # Subtract mean
        if blockwise_mean:
            block_mean = np.mean(block, axis=0)
        else:
            block_mean = q_mean
        block_centered = block - block_mean
        # Variance normalization if requested
        if normvar:
            block_var = np.var(block_centered, axis=0, ddof=1)
            block_var[block_var < 4 * np.finfo(float).eps] = 1.0 # Avoid division by zero
            block_centered = block_centered / block_var
        # Apply window and FFT
        q_hat[:, :, iblk] = cw / nfft * fft.fft(block_centered * window_broadcast, axis=0)
    return q_hat

def spod_function(qhat, nblocks, dst, w, return_psi=False):
    """Compute SPOD modes and eigenvalues for a single frequency.
    Args:
        qhat (np.ndarray): FFT coefficients for this frequency [space, block].
        nblocks (int): Number of blocks.
        dst (float): Frequency resolution (delta f).
        w (np.ndarray): Spatial integration weights [space, 1].
        return_psi (bool): If True, also return psi (time coefficients).
    Returns:
        tuple: (phi, lambda_tilde[, psi])
            phi (np.ndarray): Spatial SPOD modes for this frequency [space, mode].
            lambda_tilde (np.ndarray): SPOD eigenvalues (energy) for this frequency [mode].
            psi (np.ndarray, optional): Time coefficients for this frequency [block, mode].
    """
    # Normalize FFT coefficients to get fluctuation matrix X_f for this frequency f.
    # Normalization ensures proper scaling for the CSD matrix.
    # The denominator represents the effective bandwidth and number of averages.
    x = qhat / np.sqrt(nblocks * dst) 
    
    # Compute the weighted cross-spectral density (CSD) matrix M_f.
    # M_f = X_f^H * W * X_f, where W is the diagonal matrix of weights.
    # This is done efficiently without forming the diagonal matrix W.
    xprime_w = np.transpose(np.conj(x)) * np.transpose(w) # Calculates X_f^H * W
    m = xprime_w @ x # Calculates (X_f^H * W) * X_f = M_f
    del xprime_w  # Free memory
    
    # Solve the eigenvalue problem: M_f * Psi_f = Psi_f * Lambda_f
    # lambda_tilde corresponds to Lambda_f (eigenvalues = modal energy)
    # psi corresponds to Psi_f (eigenvectors = temporal coefficients across blocks)
    lambda_tilde, psi = np.linalg.eigh(m) # m is Hermitian so we can use eigh instead of eig
    
    # Sort eigenvalues and eigenvectors in descending order of energy
    idx = lambda_tilde.argsort()[::-1]
    lambda_tilde = lambda_tilde[idx]
    psi = psi[:, idx]
    
    # Compute spatial SPOD modes (Phi_f) of the direct problem.
    # Phi_f = X_f * Psi_f * Lambda_f^(-1/2)
    # This projects the temporal structure Psi_f back onto the spatial domain via X_f,
    # normalized by the square root of the energy.
    # Simplified calculation for inverse square root of eigenvalues Lambda_f^(-1/2)
    inv_sqrt_lambda = np.zeros_like(lambda_tilde)
    mask = lambda_tilde > 1e-12 # Avoid division by zero or near-zero (numerical stability)
    inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])
    
    # Calculate modes: phi = X_f @ Psi_f @ diag(Lambda_f^(-1/2))
    phi = x @ psi @ np.diag(inv_sqrt_lambda)
    
    # Return spatial modes and their corresponding eigenvalues (energy)
    # Taking abs of lambda_tilde as eigenvalues should be real and positive.
    if return_psi:
        return phi, np.abs(lambda_tilde), psi
    return phi, np.abs(lambda_tilde)

class SPODAnalyzer:
    """Class for performing Spectral Proper Orthogonal Decomposition (SPOD) analysis.

    **Expected Input Data Structure:**

    The core analysis methods (compute_fft_blocks, perform_spod) expect the 
    primary input data (e.g., pressure, velocity) to be preprocessed into a 
    2D NumPy array `q` stored in `self.data['q']`. 

    The required format for `self.data['q']` is:
        - Shape: `(Ns, Nspatial)`
        - `Ns`: Number of time snapshots.
        - `Nspatial`: Total number of spatial points (e.g., Nx * Ny for a 2D grid).
        - The first dimension (axis 0) must represent time.
        - The second dimension (axis 1) must represent the flattened spatial domain.
        - **Crucially, the flattening of the spatial points must be consistent 
          across all time snapshots.**

    Additionally, the following need to be provided in `self.data`:
        - `x`: 1D NumPy array of coordinates for the first spatial dimension (length Nx).
        - `y`: 1D NumPy array of coordinates for the second spatial dimension (length Ny).
        - `Nx`, `Ny`: Integers representing the dimensions of the original spatial grid.
        - `dt`: Float representing the time step between snapshots.

    The `load_jetles_data` function in this script handles loading from a specific 
    HDF5 format and performs the necessary transpose and reshape. If loading from
    a different source (like CGNS), you would need to implement a similar loading 
    function that extracts the data and metadata, then reshapes the primary data 
    variable into the required `(Ns, Nspatial)` format before assigning it and 
    the other metadata to `self.data`.
    """
    
    def __init__(
        self,
        file_path,
        nfft=128,
        overlap=0.5,
        results_dir="./preprocess",  # For HDF files
        figures_dir="./figs",        # For PNG files
        blockwise_mean=False,
        normvar=False,
        window_norm='power',
        window_type='hamming',
        data_loader=None,
        spatial_weight_type='auto'
    ):
        """Initialize the SPOD analyzer.

        Args:
            file_path (str): Path to the HDF5 data file.
            nfft (int): Number of snapshots per FFT block.
            overlap (float): Overlap fraction between blocks (0 to < 1).
            results_dir (str): Directory to save numerical results.
            figures_dir (str): Directory to save plots.
            blockwise_mean (bool): If True, use blockwise mean subtraction.
            normvar (bool): If True, normalize by variance.
            window_norm (str): Normalization for window function ('power' or 'amplitude').
            window_type (str): Type of window ('hamming', 'hann', 'rectangular').
            data_loader (callable, optional): Custom data loading function.
            spatial_weight_type (str): Type of spatial weighting ('polar', 'uniform', 'auto').
        """
        self.file_path = file_path
        self.nfft = nfft
        self.overlap = overlap
        self.output_path = results_dir  # For backward compatibility
        self.figures_path = figures_dir  # For backward compatibility
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.blockwise_mean = blockwise_mean
        self.normvar = normvar
        self.window_norm = window_norm
        self.window_type = window_type
        self.data_loader = data_loader or load_jetles_data
        self.spatial_weight_type = spatial_weight_type
        self.data = {}
        self.phi = np.array([])
        self.lambda_values = np.array([])
        self.frequencies = np.array([])
        self.W = np.array([])
        self.fs = 0.0
        self.nblocks = 0
        
        # Ensure output directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Extract root name for output files (e.g., 'cavityPIV' from './cavityPIV.mat')
        base = os.path.basename(file_path)
        self.data_root = re.sub(r"\.[^.]*$", "", base)
        
        # Initialize parameters that will be computed during analysis
        self.data = {}           # Dictionary to hold loaded data (q, x, y, dt, etc.)
        self.W = np.array([])    # Spatial integration weights
        self.novlap = int(overlap * nfft) # Number of overlapping snapshots
        self.nblocks = 0         # Number of blocks for FFT
        self.fs = 0.0            # Sampling frequency (1/dt)
        self.St = np.array([])   # Strouhal number vector
        self.dst = 0.0           # Strouhal resolution (delta St)
        self.qhat = np.array([]) # Blocked FFT coefficients [frequency, space, block]
        self.lambda_values = np.array([]) # SPOD eigenvalues [frequency, mode]
        self.phi = np.array([])  # SPOD spatial modes [frequency, space, mode]
        self.psi = np.array([])  # Time coefficients [frequency, block, mode]
        self.L = 1.0
        self.U = 1.0
    
    def load_and_preprocess(self):
        """Load data, calculate spatial weights, and SPOD parameters."""
        # Load data (q, x, y, dt, etc.) from HDF5 file
        self.data = self.data_loader(self.file_path)
        
        # Choose spatial weights
        if self.spatial_weight_type == 'auto':
            if 'cavity' in self.file_path.lower():
                # Cavity: use uniform weights (rectangular grid)
                self.W = calculate_uniform_weights(self.data['x'], self.data['y'])
                print("Cavity case: Using uniform spatial weights (rectangular grid).")
            else:
                # Jet: use polar weights (cylindrical grid)
                self.W = calculate_polar_weights(self.data['x'], self.data['y'])
                print("Jet/other case: Using polar (cylindrical) spatial weights.")
        elif self.spatial_weight_type == 'uniform':
            self.W = calculate_uniform_weights(self.data['x'], self.data['y'])
            print("Spatial weights: uniform (rectangular grid).")
        elif self.spatial_weight_type == 'polar':
            self.W = calculate_polar_weights(self.data['x'], self.data['y'])
            print("Spatial weights: polar (cylindrical grid).")
        else:
            raise ValueError(f"Unknown spatial_weight_type: {self.spatial_weight_type}")
        
        # Set normalization constants for Strouhal number
        if 'cavity' in self.file_path.lower():
            self.L = 0.0381  # [m], cavity length
            self.U = 230.0   # [m/s], free-stream velocity
            print(f"Cavity case detected: Using L={self.L} m, U={self.U} m/s for Strouhal normalization.")
        else:
            self.L = 1.0
            self.U = 1.0
            print("Jet case or unknown: Using L=1, U=1 for Strouhal normalization.")
        
        # Calculate derived SPOD parameters
        self.nblocks = int(np.ceil((self.data['Ns'] - self.novlap) / (self.nfft - self.novlap)))
        self.fs = 1 / self.data['dt']
        f = np.linspace(0, self.fs - self.fs/self.nfft, self.nfft)
        St = f * self.L / self.U
        self.St = St[0 : self.nfft // 2 + 1]  # One-sided Strouhal vector (normalized)
        self.dst = self.St[1] - self.St[0] 
        self.strouhal = St  # Store full Strouhal vector if needed
        print("SPOD Parameters:")
        print(f"Number of snapshots: {self.data['Ns']}, Block size: {self.nfft}, "
              f"Overlap: {self.overlap} ({self.novlap} points), Number of blocks: {self.nblocks}")
        print(f"Strouhal sampling (max St): {self.St[-1]:.4f}, dSt: {self.dst:.4f}")
    
    def compute_fft_blocks(self):
        """Compute blocked FFT using the blocksfft function."""
        if 'q' not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
        
        print(f"Computing FFT with {self.nblocks} blocks...")
        self.qhat = blocksfft(
            self.data['q'], self.nfft, self.nblocks, self.novlap,
            blockwise_mean=self.blockwise_mean,
            normvar=self.normvar,
            window_norm=self.window_norm,
            window_type=self.window_type
        )
        print("FFT computation complete.")
    
    def perform_spod(self):
        """Perform SPOD analysis (eigenvalue decomposition) for each frequency."""
        if self.qhat.size == 0:
            raise ValueError("FFT blocks not computed. Call compute_fft_blocks() first.")
        
        start_time = time.time()
        # Total number of spatial points
        nq = self.data['Nx'] * self.data['Ny']
        # Number of frequencies to compute (only positive frequencies 0 to fs/2)
        n_freq = self.nfft // 2 + 1
        
        # Initialize arrays to store results
        # Eigenvalues (energy) for each mode at each frequency
        self.lambda_values = np.zeros((n_freq, self.nblocks))
        # Spatial modes for each mode at each frequency
        self.phi = np.zeros((n_freq, nq, self.nblocks), dtype=complex)
        # Time coefficients for each mode at each frequency
        self.psi = np.zeros((n_freq, self.nblocks, self.nblocks), dtype=complex)
        
        print("Performing SPOD for each frequency...")
        # Compute SPOD for each frequency f in the one-sided spectrum (0 to fs/2)
        for i in range(n_freq):
            # Extract FFT data for the current frequency i
            # qhat has shape [frequency, space, block]
            # We need [space, block] for spod_function
            qhat_freq = self.qhat[i, :, :]
            
            # Call the core SPOD function for this frequency
            phi_freq, lambda_freq, psi_freq = spod_function(
                qhat_freq, self.nblocks, self.dst, self.W, return_psi=True
            )
            
            # Store results
            self.phi[i, :, :] = phi_freq
            self.lambda_values[i, :] = lambda_freq
            self.psi[i, :, :] = psi_freq
            
            # Print progress (optional)
            if (i + 1) % 10 == 0 or i == n_freq - 1:
                 print(f"  Processed frequency {i+1}/{n_freq} (St = {self.St[i]:.4f})")
        
        print(f"SPOD eigenvalue decomposition completed in {time.time() - start_time:.2f} seconds")
    
    def save_results(self):
        """Save SPOD modes, eigenvalues, frequencies, and parameters to an HDF5 file."""
        if self.phi.size == 0 or self.lambda_values.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Construct full path for the output file
        save_name = f"{self.data_root}_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots.hdf5"
        save_path = os.path.join(self.results_dir, save_name)
        print(f"Saving results to {save_path}")
        
        with h5py.File(save_path, "w") as fsnap:
            # Create datasets for modes, eigenvalues, frequencies, and grid
            # Using gzip compression for efficiency
            fsnap.create_dataset("Phi", data=self.phi, compression="gzip") # Modes [freq, space, mode]
            fsnap.create_dataset("Lambda", data=self.lambda_values, compression="gzip") # Eigenvalues [freq, mode]
            fsnap.create_dataset("St", data=self.St, compression="gzip") # One-sided Strouhal vector [freq]
            fsnap.create_dataset("x", data=self.data['x'], compression="gzip") # Axial coordinates
            fsnap.create_dataset("y", data=self.data['y'], compression="gzip") # Radial coordinates

            # Save key scalar parameters as attributes or small datasets
            fsnap.attrs["Nfft"] = self.nfft
            fsnap.attrs["overlap"] = self.overlap
            fsnap.attrs["Ns"] = self.data['Ns']
            fsnap.attrs["fs"] = self.fs
            fsnap.attrs["nblocks"] = self.nblocks
            fsnap.attrs["dt"] = self.data['dt']
    
    def plot_eigenvalues(self, n_modes=10, highlight_St=None):
        """Plot the SPOD eigenvalue spectrum (energy vs. Strouhal number) for leading modes.
        Optionally highlight a specific St value (e.g., the selected mode peak)."""
        if self.lambda_values.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        
        print("Plotting SPOD eigenvalues...")
        
        plt.figure(figsize=(10, 6))
        plt.rc("text", usetex=False)
        plt.rc("font", family="serif", size=12)
        
        n_modes_to_plot = min(n_modes, self.lambda_values.shape[1])
        for i in range(n_modes_to_plot):
            plt.loglog(self.St, self.lambda_values[:, i], 
                       label=f"Mode {i+1}", marker='o', markersize=3, linestyle='-')
        
        # Highlight the selected St peak if provided
        if highlight_St is not None:
            idx = np.argmin(np.abs(self.St - highlight_St))
            plt.scatter(self.St[idx], self.lambda_values[idx, 0],
                        color='red', s=80, edgecolor='k', zorder=10, label=f"Peak St={self.St[idx]:.3f}")
        
        plt.legend()
        plt.xlabel(r"Strouhal number (St)")
        plt.ylabel(r"SPOD Eigenvalue $\lambda$")
        plt.title(r"SPOD Eigenvalue Spectrum vs. St")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        # Ensure figures directory exists
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_eigenvalues_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots.png"
        plt.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Eigenvalue plot saved to {filename}")
    
    def plot_modes(self, st_target, n_modes=4):
        """Plot the real part of spatial SPOD modes (Phi) for a target Strouhal number."""
        if self.phi.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        
        # Find the index (st_idx) of the Strouhal number in self.St closest to st_target
        st_idx = np.argmin(np.abs(self.St - st_target))
        st_value = self.St[st_idx]
        print(f"Plotting SPOD modes for St ≈ {st_value:.4f} (target: {st_target:.4f})...", flush=True)
        
        # Setup grid layout for subplots
        n_modes_to_plot = min(n_modes, self.phi.shape[2], 4)

        if n_modes_to_plot == 0:
            print("  Warning: No modes available to plot.")
            return
            
        # Set layout to 2x2 grid
        nrows = 2
        ncols = 2

        # Create figure and axes
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False, constrained_layout=True)
        axes = axes.flatten()
        
        plt.rc("text", usetex=False)
        plt.rc("font", family="serif", size=10)
        cmap = plt.get_cmap("bwr")
        
        # Plot each requested mode
        max_abs_val = 0
        phi_modes_real = []
        for i in range(n_modes_to_plot):
             phi_real = self.phi[st_idx, :, i].real
             phi_2d = np.reshape(phi_real, (self.data['Nx'], self.data['Ny'])).T
             phi_modes_real.append(phi_2d)
             max_abs_val = max(max_abs_val, np.max(np.abs(phi_real)))
        
        interval = max_abs_val * 1.0
        levels = np.linspace(-interval, interval, 61)

        for i in range(n_modes_to_plot):
            phi_2d = phi_modes_real[i]
            ax = axes[i]
            im = ax.contourf(
                self.data['x'], self.data['y'], phi_2d, 
                levels=levels, cmap=cmap, 
                vmin=-interval, vmax=interval, 
                extend='both'
            )
            ax.set_aspect("equal")
            ax.set_title(f"Mode {i+1}", size=12)
            if i >= (nrows - 1) * ncols:
                 ax.set_xlabel("$x$", fontsize=12)
            if i % ncols == 0:
                 ax.set_ylabel("$r$", fontsize=12)
            else:
                 ax.set_yticklabels([])
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        for i in range(n_modes_to_plot, len(axes)):
            axes[i].axis('off')

        fig.colorbar(im, ax=axes[:n_modes_to_plot], shrink=0.8, label=r'Real($\Phi$)')
        fig.suptitle(rf"SPOD Modes at $St \approx {st_value:.4f}$", fontsize=14)
        
        # Ensure figures directory exists
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_modes_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots_St{st_value:.4f}.png"
        fig.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Mode plot saved to {filename}")
    
    def plot_eig_complex_plane(self, n_modes=4, st_target=None):
        """
        Plot the eigenvalues for a given Strouhal number (or the dominant one) in the complex plane:
        - x-axis: Real part
        - y-axis: Imaginary part
        Each eigenvalue corresponds to a mode at the selected frequency.
        """
        if self.lambda_values.size == 0 or self.phi.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        # Determine which St to use
        if st_target is None:
            # Use the dominant peak as in plot_modes
            peaks, _ = find_peaks(self.lambda_values[:, 0])
            valid_peaks = [i for i in peaks if self.St[i] > 0.1]
            if valid_peaks:
                idx_peak = valid_peaks[np.argmax(self.lambda_values[valid_peaks, 0])]
                st_idx = idx_peak
                st_value = self.St[st_idx]
            else:
                st_idx = np.argmax(self.lambda_values[:, 0])
                st_value = self.St[st_idx]
        else:
            st_idx = np.argmin(np.abs(self.St - st_target))
            st_value = self.St[st_idx]
        # Get eigenvalues (complex) for the selected frequency
        # For SPOD, lambda_values are real, but phi (modes) are complex
        # We'll plot the first n_modes eigenvectors' first spatial point (as a simple example)
        eigvecs = self.phi[st_idx, :, :n_modes]  # shape: (space, n_modes)
        # For each mode, plot the real vs imag part of all spatial points
        plt.figure(figsize=(6,6))
        for i in range(eigvecs.shape[1]):
            plt.scatter(eigvecs[:, i].real, eigvecs[:, i].imag, label=f"Mode {i+1}", alpha=0.7, s=18)
        plt.xlabel("Real part of Mode")
        plt.ylabel("Imaginary part of Mode")
        plt.title(rf"SPOD Mode Eigenvectors in Complex Plane ($St \approx {st_value:.4f}$)")
        plt.legend()
        plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        # Ensure figures directory exists
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_modes_complex_plane_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots_St{st_value:.4f}.png"
        plt.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  Complex plane plot saved to {filename}")
    
    def plot_time_coeffs(self, st_target=None, coeffs_idx=[0], n_blocks_plot=None):
        """
        Plot the real and imaginary parts of the time coefficients (psi) for selected modes at a given St.
        coeffs_idx: list of mode indices to plot (e.g., [0, 1])
        n_blocks_plot: if not None, limit number of blocks/time samples to plot
        """
        if not hasattr(self, 'psi') or self.psi.size == 0:
            print("No time coefficients found. Run SPOD first.")
            return
        if st_target is None:
            # Use dominant peak as in plot_modes
            peaks, _ = find_peaks(self.lambda_values[:, 0])
            valid_peaks = [i for i in peaks if self.St[i] > 0.1]
            if valid_peaks:
                st_idx = valid_peaks[np.argmax(self.lambda_values[valid_peaks, 0])]
                st_value = self.St[st_idx]
            else:
                st_idx = np.argmax(self.lambda_values[:, 0])
                st_value = self.St[st_idx]
        else:
            st_idx = np.argmin(np.abs(self.St - st_target))
            st_value = self.St[st_idx]
        time_coeffs = self.psi[st_idx, :, :]  # shape: (nblocks, nblocks)
        if n_blocks_plot is not None:
            time_coeffs = time_coeffs[:n_blocks_plot, :]
        plt.figure(figsize=(8, 5))
        for idx in coeffs_idx:
            plt.plot(time_coeffs[:, idx].real, label=f"Mode {idx+1} (real)")
            plt.plot(time_coeffs[:, idx].imag, '--', label=f"Mode {idx+1} (imag)")
        plt.xlabel("Block index (time)")
        plt.ylabel("Time coefficient")
        plt.title(rf"SPOD Time Coefficients at $St \approx {st_value:.4f}$")
        plt.legend()
        plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        # Ensure figures directory exists
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_time_coeffs_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots_St{st_value:.4f}.png"
        plt.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  Time coefficients plot saved to {filename}")
    
    def run_analysis(self, plot_st_target=None, plot_n_modes_eig=10, plot_n_modes_spatial=4):
        print("Starting SPOD analysis...")
        start_total_time = time.time()
        
        self.load_and_preprocess()
        self.compute_fft_blocks()
        self.perform_spod()
        self.save_results()
        
        print("\nGenerating visualizations...")
        highlight_St = None

        if plot_st_target is None:
            # Find all local maxima (peaks) in mode 1
            peaks, _ = find_peaks(self.lambda_values[:, 0])
            # Filter peaks to those with St > 0.1
            valid_peaks = [i for i in peaks if self.St[i] > 0.1]
            if valid_peaks:
                # Take the peak with the highest eigenvalue among valid peaks
                idx_peak = valid_peaks[np.argmax(self.lambda_values[valid_peaks, 0])]
                plot_st_target = self.St[idx_peak]
                highlight_St = plot_st_target
                print(f"No target St provided, plotting modes for dominant PEAK St > 0.1: {plot_st_target:.4f}")
            else:
                # Fallback: just use the global maximum
                dominant_freq_idx = np.argmax(self.lambda_values[:, 0])
                plot_st_target = self.St[dominant_freq_idx]
                highlight_St = plot_st_target
                print(f"No St > 0.1 found, plotting modes for dominant St: {plot_st_target:.4f}")
        else:
            highlight_St = plot_st_target

        self.plot_eigenvalues(n_modes=min(plot_n_modes_eig, self.nblocks), highlight_St=highlight_St)
        self.plot_modes(st_target=plot_st_target, n_modes=min(plot_n_modes_spatial, self.nblocks))
        self.plot_eig_complex_plane(n_modes=min(plot_n_modes_spatial, self.nblocks), st_target=plot_st_target)
        self.plot_time_coeffs(st_target=plot_st_target)
        end_total_time = time.time()
        print(f"\nSPOD analysis completed successfully in {end_total_time - start_total_time:.2f} seconds.")

# Example usage when the script is run directly
if __name__ == "__main__":
    # --- Configuration ---
    data_file = "./data/jetLES_small.mat" # Updated data path
    #data_file = "./data/jetLES.mat" # Path to your data file
    #data_file = "./data/cavityPIV.mat" # Path to your data file

    results_dir = "./preprocess"   # Directory for HDF5 results
    figures_dir = "./figs"          # Directory for plots
    # Default parameters
    nfft_param = 128                 # FFT block size
    overlap_param = 0.5              # Overlap fraction (50%)
    # Optional: Specify a frequency target for mode plots
    # If None, it will plot modes at the frequency with peak energy in Mode 1.
    freq_target_for_plots = None
    # ---------------------
    # Set case-specific parameters for cavity (to match MATLAB reference)
    if 'cavity' in data_file.lower():
        nfft_param = 256
        overlap_param = 128 / 256  # 0.5
        window_type_param = 'sine'  # Sine window for cavity, as in MATLAB
        spatial_weight_type_param = 'uniform'  # Rectangular grid for cavity
        print("Cavity case detected: Using nfft=256, overlap=128 (50%), sine window to match MATLAB reference.")
    elif 'jet' in data_file.lower():
        window_type_param = 'hamming'  # Default for jet case
        spatial_weight_type_param = 'polar'  # Cylindrical grid for jet
        print("Jet case detected: Using default nfft, overlap, and Hamming window.")
    else:
        window_type_param = 'hamming'
        spatial_weight_type_param = 'uniform'
        print("Unknown case: Using default nfft, overlap, and Hamming window.")
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at '{data_file}'")
        exit()

    # Set case-specific data loader
    if 'cavity' in data_file.lower():
        data_loader_param = load_mat_data
        print("Cavity case detected: Using load_mat_data.")
    elif 'jet' in data_file.lower():
        data_loader_param = load_jetles_data
        print("Jet case detected: Using load_jetles_data.")
    else:
        data_loader_param = load_mat_data
        print("Unknown case: Using load_mat_data.")

    # Create SPOD analyzer instance
    spod_analyzer = SPODAnalyzer(
        file_path=data_file,
        nfft=nfft_param,
        overlap=overlap_param,
        results_dir=results_dir,
        figures_dir=figures_dir,
        window_type=window_type_param,
        data_loader=data_loader_param,
        spatial_weight_type=spatial_weight_type_param
    )
    
    # Run the full analysis and plotting pipeline
    spod_analyzer.run_analysis(
        plot_st_target=freq_target_for_plots, 
        plot_n_modes_eig=10,       # Number of modes in eigenvalue plot
        plot_n_modes_spatial=4     # Number of modes in spatial plot
    )
