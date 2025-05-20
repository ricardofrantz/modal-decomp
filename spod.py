#!/usr/bin/env python3
"""
Extract modes with Spectral Proper Orthogonal Decomposition (SPOD)

Author: R. Frantz

Reference codes:
    - https://github.com/SpectralPOD/spod_matlab/tree/master
    - https://github.com/MathEXLab/PySPOD/blob/main/tutorials/tutorial1/tutorial1.ipynb
"""
# All core imports and configs are available via utils
from utils import *

class SPODAnalyzer(BaseAnalyzer):
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
    
    def __init__(self,
                 file_path,
                 nfft=128,
                 overlap=0.5,
                 results_dir=RESULTS_DIR,
                 figures_dir=FIGURES_DIR,
                 blockwise_mean=False,
                 normvar=False,
                 window_norm='power',
                 window_type='hamming',
                 data_loader=None,
                 spatial_weight_type='auto'):
        super().__init__(file_path=file_path,
                         nfft=nfft,
                         overlap=overlap,
                         results_dir=results_dir,
                         figures_dir=figures_dir,
                         data_loader=data_loader,
                         spatial_weight_type=spatial_weight_type)
        self.blockwise_mean = blockwise_mean
        self.normvar = normvar
        self.window_norm = window_norm
        self.window_type = window_type
        # SPOD-specific fields
        self.phi = np.array([])
        self.lambda_values = np.array([])
        self.frequencies = np.array([])
        self.psi = np.array([])
        self.St = np.array([])
        self.dst = 0.0
        self.L = 1.0
        self.U = 1.0
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
    
    def load_and_preprocess(self):
        super().load_and_preprocess()
        # Set normalization constants for Strouhal number
        if 'cavity' in self.file_path.lower():
            self.L = 0.0381
            self.U = 230.0
            print(f"Cavity case detected: Using L={self.L} m, U={self.U} m/s for Strouhal normalization.")
        else:
            self.L = 1.0
            self.U = 1.0
            print("Jet case or unknown: Using L=1, U=1 for Strouhal normalization.")
        # Calculate Strouhal vector
        self.fs = 1 / self.data['dt']
        f = np.linspace(0, self.fs - self.fs/self.nfft, self.nfft)
        St = f * self.L / self.U
        self.St = St[0 : self.nfft // 2 + 1]
        self.dst = self.St[1] - self.St[0]
        self.strouhal = St
    
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
        if self.phi.size == 0 or self.lambda_values.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        save_name = f"{self.data_root}_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots.hdf5"
        save_path = os.path.join(self.results_dir, save_name)
        print(f"Saving results to {save_path}")
        with h5py.File(save_path, "w") as fsnap:
            fsnap.create_dataset("Phi", data=self.phi, compression="gzip")
            fsnap.create_dataset("Lambda", data=self.lambda_values, compression="gzip")
            fsnap.create_dataset("St", data=self.St, compression="gzip")
            fsnap.create_dataset("x", data=self.data['x'], compression="gzip")
            fsnap.create_dataset("y", data=self.data['y'], compression="gzip")
            fsnap.attrs["Nfft"] = self.nfft
            fsnap.attrs["overlap"] = self.overlap
            fsnap.attrs["Ns"] = self.data['Ns']
            fsnap.attrs["fs"] = self.fs
            fsnap.attrs["nblocks"] = self.nblocks
            fsnap.attrs["dt"] = self.data['dt']
    
    def plot_eigenvalues(self, n_modes=10, highlight_St=None):
        """Plot the SPOD eigenvalue spectrum (energy vs. St) for leading modes.
        Optionally highlight a specific St value (e.g., the selected mode peak)."""
        if self.lambda_values.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        print("Plotting SPOD eigenvalues...")
        plt.figure(figsize=(10, 6))
        plt.rc("text", usetex=USE_LATEX)
        plt.rc("font", family=FONT_FAMILY, size=FONT_SIZE)
        n_modes_to_plot = min(n_modes, self.lambda_values.shape[1])
        for i in range(n_modes_to_plot):
            plt.loglog(self.St, self.lambda_values[:, i],
                       label=f"Mode {i+1}", marker='o', markersize=3, linestyle='-')
        if highlight_St is not None:
            idx = np.argmin(np.abs(self.St - highlight_St))
            plt.scatter(self.St[idx], self.lambda_values[idx, 0],
                        color='red', s=80, edgecolor='k', zorder=10, label=f"Peak St={self.St[idx]:.3f}")
        plt.legend()
        plt.xlabel(r"St")
        plt.ylabel(r"SPOD Eigenvalue $\lambda$")
        plt.title(r"SPOD Eigenvalue Spectrum vs. St")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_eigenvalues_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots.{FIG_FORMAT}"
        plt.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=FIG_DPI, format=FIG_FORMAT)
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
