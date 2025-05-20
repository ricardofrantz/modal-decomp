#!/usr/bin/env python3
"""
Extract coherent bispectral modes with BiSpectral Mode Decomposition (BSMD)

Reference: "Bispectral mode decomposition of nonlinear flows."  Schmidt, O. T. (2020).

Definitions:
  bispectrum B(f1,f2) = ⟨ X(f1) X(f2) X*(f1+f2) ⟩,
  triad (f1,f2,f3) satisfying f1 + f2 = f3.

Method:
  1. Compute FFT blocks via Welch’s method: qhat[f, j, b].
  2. For each triad, form:
       A_jb = conj[ qhat[p1, j, b] · qhat[p2, j, b] ],
       B_jb =     qhat[p3, j, b].
  3. Build bispectral correlation:
       C = A^H W B,  C_{bb'} = Σ_j A_jb^* W_j B_jb'.
  4. Solve: C a = λ a, obtain eigenmodes a.
  5. Spatial modes:
       Φ1_j = Σ_b a_b^* B_jb,  Φ2_j = Σ_b a_b^* A_jb.
"""
# All core imports and configs are available via utils
from utils import *

# Standard static triad list
ALL_TRIADS = [
    (8, -8,  0),(7, -7,  0),(8, -7,  1),(6, -6,  0),(7, -6,  1),(8, -6,  2),
    (5, -5,  0),(6, -5,  1),(7, -5,  2),(8, -5,  3),(4, -4,  0),(5, -4,  1),
    (6, -4,  2),(7, -4,  3),(8, -4,  4),(3, -3,  0),(4, -3,  1),(5, -3,  2),
    (6, -3,  3),(7, -3,  4),(8, -3,  5),(2, -2,  0),(3, -2,  1),(4, -2,  2),
    (5, -2,  3),(6, -2,  4),(7, -2,  5),(8, -2,  6),(1, -1,  0),(2, -1,  1),
    (3, -1,  2),(4, -1,  3),(5, -1,  4),(6, -1,  5),(7, -1,  6),(8, -1,  7),
    (0,  0,  0),(1,  0,  1),(2,  0,  2),(3,  0,  3),(4,  0,  4),(5,  0,  5),
    (6,  0,  6),(7,  0,  7),(8,  0,  8),(1,  1,  2),(2,  1,  3),(3,  1,  4),
    (4,  1,  5),(5,  1,  6),(6,  1,  7),(7,  1,  8),(2,  2,  4),(3,  2,  5),
    (4,  2,  6),(5,  2,  7),(6,  2,  8),(3,  3,  6),(4,  3,  7),(5,  3,  8),
    (4,  4,  8)
]

class BSMDAnalyzer(BaseAnalyzer):
    """
    Class for performing Bispectral SPOD (BSMD) analysis.
    """
    def __init__(self,
                 file_path,
                 nfft=128,
                 overlap=0.5,
                 results_dir="./preprocess",
                 figures_dir="./figs",
                 data_loader=None,
                 spatial_weight_type='auto',
                 use_static_triads=True,
                 static_triads=ALL_TRIADS):
        super().__init__(file_path=file_path,
                         nfft=nfft,
                         overlap=overlap,
                         results_dir=results_dir,
                         figures_dir=figures_dir,
                         data_loader=data_loader,
                         spatial_weight_type=spatial_weight_type)
        self.use_static_triads = use_static_triads
        self.static_triads = static_triads
        self.triads = []
        self.a = np.array([])
        self.lambda_vals = np.array([])
        self.Phi1 = np.array([])
        self.Phi2 = np.array([])
        """
        Initialize the BSMD analyzer.

        Args:
            file_path (str): Path to the HDF5 or .mat data file.
            nfft (int): Number of snapshots per FFT block.
            overlap (float): Overlap fraction between blocks (0–1).
            results_dir (str): Directory for HDF5 results.
            figures_dir (str): Directory for figures.
            data_loader (callable): Function to load data (default: load_mat_data).
            spatial_weight_type (str): 'auto', 'uniform', or 'polar'.
            use_static_triads (bool): Use predefined static triads (default: True).
            static_triads (list): Predefined static triads (default: ALL_TRIADS).
        """
        self.file_path = file_path
        self.nfft = nfft
        self.overlap = overlap
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.data_loader = data_loader or load_mat_data
        self.spatial_weight_type = spatial_weight_type
        self.use_static_triads = use_static_triads
        self.static_triads = static_triads
        
        # Ensure output directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        # Derive base name for outputs
        base = os.path.basename(file_path)
        self.data_root = re.sub(r"\.[^.]*$", "", base)

        # Placeholders
        self.data = {}
        self.W = np.array([])
        self.novlap = int(overlap * nfft)
        self.nblocks = 0
        self.fs = 0.0
        self.qhat = np.array([])
        self.triads = []
        self.a = np.array([])
        self.lambda_vals = np.array([])
        self.Phi1 = np.array([])
        self.Phi2 = np.array([])

    def load_and_preprocess(self):
        super().load_and_preprocess()

    def compute_fft_blocks(self):
        super().compute_fft_blocks()

    def perform_BSMD(self):
        """
        Perform BiSpectral Mode Decomposition (BSMD).

        Bispectrum B(f1,f2) = ⟨ X(f1) X(f2) X*(f1+f2) ⟾ over blocks.

        For triad (p1,p2,p3):
            A_jb = conj[ qhat[p1,j,b] * qhat[p2,j,b] ]
            B_jb =     qhat[p3,j,b]
        C = A^H W B constructs bispectral correlations:
            C_{bb'} = Σ_j A_jb^* W_j B_jb'.
        Solve C a = λ a and normalize a.

        Spatial bispectral modes:
            Φ1_j = Σ_b a_b^* B_jb
            Φ2_j = Σ_b a_b^* A_jb
        """
        # Choose triad set: static or full dynamic
        if self.use_static_triads and self.static_triads:
            triads = self.static_triads
        else:
            n_freq = self.qhat.shape[0]
            triads = [(p1, p2, p1+p2)
                      for p1 in range(n_freq)
                      for p2 in range(-n_freq+1, n_freq)
                      if 0 <= p1+p2 < n_freq]
        self.triads = triads
        NT = len(triads)
        nb = self.nblocks
        nq = self.data['Nx'] * self.data['Ny']
        W_flat = self.W.flatten()
        # Allocate outputs
        self.lambda_vals = np.zeros(NT, dtype=complex)
        self.a = np.zeros((NT, nb), dtype=complex)
        self.Phi1 = np.zeros((NT, nq), dtype=complex)
        self.Phi2 = np.zeros((NT, nq), dtype=complex)

        print(f"Running BSMD on {NT} triads, {nb} blocks...")
        for ti, (p1, p2, p3) in enumerate(triads):
            # Build cross-frequency objects
            if p2 >= 0:
                A = np.conj(self.qhat[p1] * self.qhat[p2])
            else:
                A = np.conj(self.qhat[p1]) * self.qhat[-p2]
            B = self.qhat[p3]
            # Bispectral density matrix C (block x block)
            C = A.T.dot(B * W_flat[:, None])
            # Eigen-decomposition
            vals, vecs = eig(C)
            idx = np.argmax(np.abs(vals))
            a = vecs[:, idx]
            a /= np.linalg.norm(a)
            self.a[ti] = a
            self.lambda_vals[ti] = vals[idx]
            # Spatial bispectral modes
            self.Phi1[ti] = (a.conj()[None, :] * B).sum(axis=1)
            if p2 >= 0:
                self.Phi2[ti] = (a.conj()[None, :] * (self.qhat[p1] * self.qhat[p2])).sum(axis=1)
            else:
                self.Phi2[ti] = (a.conj()[None, :] * (self.qhat[p1] * np.conj(self.qhat[-p2]))).sum(axis=1)
        print("BSMD analysis completed.")

    def save_results(self, fname=None):
        """Save triads, eigenvalues, modes, and weights to HDF5."""
        if fname is None:
            fname = f"{self.data_root}_BSMD.h5"
        # Ensure output directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        fp = os.path.join(self.results_dir, fname)
        with h5py.File(fp, 'w') as f:
            f.create_dataset('triads', data=np.array(self.triads))
            f.create_dataset('lambda_real', data=self.lambda_vals.real)
            f.create_dataset('lambda_imag', data=self.lambda_vals.imag)
            f.create_dataset('a_real', data=self.a.real)
            f.create_dataset('a_imag', data=self.a.imag)
            f.create_dataset('Phi1', data=self.Phi1)
            f.create_dataset('Phi2', data=self.Phi2)
            f.create_dataset('x', data=self.data['x'])
            f.create_dataset('y', data=self.data['y'])
            f.create_dataset('W', data=self.W)
        print(f"Results saved to {fp}")

    def run_analysis(self):
        """Execute the full BSMD pipeline."""
        start = time.time()
        self.load_and_preprocess()
        self.compute_fft_blocks()
        self.perform_BSMD()
        self.save_results()
        print(f"Total BSMD runtime: {time.time()-start:.2f} s")

if __name__ == '__main__':
    # Example usage
    data_file = './data/jetLES.mat'  # Updated data path
    # Choose loader based on file
    if 'jet' in data_file.lower():
        loader = load_jetles_data
        spatial_weight = 'polar'
    else:
        loader = load_mat_data
        spatial_weight = 'uniform'

    analyzer = BSMDAnalyzer(
        file_path=data_file,
        nfft=128,
        overlap=0.5,
        results_dir='./preprocess',  # HDF files go here
        figures_dir='./figs',       # PNG files go here
        data_loader=loader,
        spatial_weight_type=spatial_weight,
        use_static_triads=True,
        static_triads=ALL_TRIADS,
    )
    analyzer.run_analysis()

    # **Plot 1: BSMD Eigenvalue Magnitudes**
    lambdas = np.abs(analyzer.lambda_vals)
    plt.figure()
    plt.plot(lambdas, 'o-')
    plt.xlabel('Triad index')
    plt.ylabel('Eigenvalue magnitude')
    plt.title('BSMD eigenvalue magnitudes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./figs', f"{analyzer.data_root}_BSMD_eigenvalues.png"))
    plt.close()

    # **Additional Figures per Schmidt (2020)**

    # **Plot 2: Complex Eigenvalue Plane**
    vals = analyzer.lambda_vals
    plt.figure()
    plt.scatter(np.real(vals), np.imag(vals), marker='o')
    plt.xlabel('Real(λ)')
    plt.ylabel('Imag(λ)')
    plt.title('BSMD eigenvalues (complex plane)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./figs', f"{analyzer.data_root}_BSMD_eig_complex_plane.png"))
    plt.close()

    # **Function to Translate Triad Indices to Strouhal Numbers**
    def triad_to_strouhal(triad, fs, nfft):
        """
        Convert triad indices (p_k, p_l, p_k+p_l) to Strouhal numbers (St_k, St_l, St_(k+l))
        and their decomposition in terms of St_0.

        Args:
            triad (tuple): Triad indices (p_k, p_l, p_k+p_l).
            fs (float): Sampling frequency.
            nfft (int): Number of FFT points.

        Returns:
            dict: Contains 'decomposition' (str) and 'values' (tuple of St_k, St_l, St_(k+l)).
        """
        p_k, p_l, p_k_plus_l = triad
        St_0 = fs / nfft  # Base Strouhal number (frequency resolution)
        St_k = p_k * St_0
        St_l = p_l * St_0
        St_k_plus_l = p_k_plus_l * St_0
        decomposition = f"({p_k}*St_0, {p_l}*St_0, {p_k_plus_l}*St_0)"
        values = (St_k, St_l, St_k_plus_l)
        return {'decomposition': decomposition, 'values': values}

    # **Plot 3: Spatial Bispectral Modes for Top 5 Dominant Triads**
    # Find the indices of the top 5 dominant triads based on |λ|
    lambda_mags = np.abs(analyzer.lambda_vals)
    top_5_indices = np.argsort(lambda_mags)[-5:][::-1]  # Top 5 in descending order
    top_5_triads = [analyzer.triads[idx] for idx in top_5_indices]

    Nx, Ny = analyzer.data['Nx'], analyzer.data['Ny']
    x = analyzer.data['x']
    y = analyzer.data['y']
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Create a 5x2 subplot grid (5 rows for each triad, 2 columns for Phi1 and Phi2)
    fig, axs = plt.subplots(5, 2, figsize=(12, 20), sharex=True, sharey=True)
    for i, (idx, triad) in enumerate(zip(top_5_indices, top_5_triads)):
        phi1 = analyzer.Phi1[idx].reshape(Nx, Ny)
        phi2 = analyzer.Phi2[idx].reshape(Nx, Ny)
        # Convert triad to Strouhal numbers and decomposition
        result = triad_to_strouhal(triad, analyzer.fs, analyzer.nfft)
        decomposition = result['decomposition']
        St_k, St_l, St_k_plus_l = result['values']
        # Plot Phi1 (left column)
        pcm1 = axs[i, 0].pcolormesh(Y, X, np.real(phi1), shading='auto')
        fig.colorbar(pcm1, ax=axs[i, 0])
        axs[i, 0].set_title(f"Phi1 real part\n{decomposition}\n(St_1={St_k:.3f}, St_2={St_l:.3f}, St_3={St_k_plus_l:.3f})")
        axs[i, 0].set_ylabel('x')
        # Plot Phi2 (right column)
        pcm2 = axs[i, 1].pcolormesh(Y, X, np.real(phi2), shading='auto')
        fig.colorbar(pcm2, ax=axs[i, 1])
        axs[i, 1].set_title(f"Phi2 real part\n{decomposition}\n(St_1={St_k:.3f}, St_2={St_l:.3f}, St_3={St_k_plus_l:.3f})")
        # Set labels for the bottom row
        if i == 4:
            axs[i, 0].set_xlabel('y')
            axs[i, 1].set_xlabel('y')
    plt.tight_layout()
    fig.savefig(os.path.join('./figs', f"{analyzer.data_root}_BSMD_modes_top_5_triads.png"))
    plt.close(fig)

    # **Plot 4: BSMD Eigenvalue Magnitudes in (St_1, St_2) Plane with Full and Zoomed-In Views**
    # Extract frequencies and eigenvalue magnitudes
    triads_arr = np.array(analyzer.triads)
    fs = analyzer.fs
    nfft = analyzer.nfft
    St1_vals = triads_arr[:, 0] * fs / nfft
    St2_vals = triads_arr[:, 1] * fs / nfft
    lambda_mags = np.abs(analyzer.lambda_vals)

    # Compute logarithmic magnitudes
    log_lambda_mags = np.log10(lambda_mags + 1e-20)  # Add small value to avoid log(0)

    # Dynamically determine frequency ranges for the full plot
    St1_max = np.max(St1_vals)
    St2_min = np.min(St2_vals)
    St2_max = np.max(St2_vals)

    # Estimate fundamental frequency St0 from triad with p2=0 and maximum |λ|
    idx_p2_zero = np.where(triads_arr[:, 1] == 0)[0]
    if len(idx_p2_zero) > 0:
        idx_max = idx_p2_zero[np.argmax(lambda_mags[idx_p2_zero])]
        St0 = St1_vals[idx_max]
    else:
        St0 = fs / nfft  # Default to frequency resolution if no p2=0 triad exists

    # Set minimum zoom range to prevent identical axis limits
    min_zoom = fs / nfft
    k = 4  # Multiplier for zoom range, adjustable if needed
    zoom_St1_max = max(k * St0, min_zoom)
    zoom_St2_min = -zoom_St1_max
    zoom_St2_max = zoom_St1_max

    # Create grids for full and zoomed plots
    n_St1 = 100  # Number of grid points for St_1
    n_St2 = 160  # Number of grid points for St_2
    St1_grid = np.linspace(0, St1_max, n_St1)
    St2_grid = np.linspace(St2_min, St2_max, n_St2)
    ST1, ST2 = np.meshgrid(St1_grid, St2_grid)

    St1_zoom_grid = np.linspace(0, zoom_St1_max, n_St1)
    St2_zoom_grid = np.linspace(zoom_St2_min, zoom_St2_max, n_St2)
    ST1_zoom, ST2_zoom = np.meshgrid(St1_zoom_grid, St2_zoom_grid)

    # Initialize grid arrays with NaN for regions outside the triangular region
    log_lambda_grid = np.full(ST1.shape, np.nan)
    log_lambda_zoom = np.full(ST1_zoom.shape, np.nan)

    # Map log_lambda_mags to the grids
    for St1, St2, log_mag in zip(St1_vals, St2_vals, log_lambda_mags):
        if St1 >= 0 and St1 + St2 >= 0:  # Triangular region condition
            i_St1 = np.argmin(np.abs(St1_grid - St1))
            i_St2 = np.argmin(np.abs(St2_grid - St2))
            log_lambda_grid[i_St2, i_St1] = log_mag
            # Map to zoom grid if within zoom range
            if 0 <= St1 <= zoom_St1_max and zoom_St2_min <= St2 <= zoom_St2_max:
                i_St1_zoom = np.argmin(np.abs(St1_zoom_grid - St1))
                i_St2_zoom = np.argmin(np.abs(St2_zoom_grid - St2))
                log_lambda_zoom[i_St2_zoom, i_St1_zoom] = log_mag

    # Set color scale dynamically
    finite_logs = log_lambda_mags[np.isfinite(log_lambda_mags)]
    if len(finite_logs) > 0:
        vmin = np.percentile(finite_logs, 1)  # 1st percentile for lower bound
        vmax = max(0, np.percentile(finite_logs, 99))  # 99th percentile, ensure ≥ 0
    else:
        vmin, vmax = -20, 0  # Default values if no finite logs

    # Ensure vmin < vmax for contour levels
    if vmin >= vmax:
        if vmin == 0:
            vmin = -1  # Arbitrary small value to allow contour levels
        else:
            vmin = vmax - 1  # Ensure a small range if they're equal

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot (a): Full Bispectrum with Colorful Contour Lines
    pcm1 = ax1.pcolormesh(ST1, ST2, log_lambda_grid, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    # Add colorful contour lines to the full bispectrum plot
    levels = np.linspace(vmin, vmax, 10)
    levels = np.sort(levels)  # Ensure levels are increasing
    # Define a list of colors for the contours (cycling through if needed)
    contour_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'cyan']
    ax1.contour(ST1, ST2, log_lambda_grid, levels=levels, colors=contour_colors, linewidths=0.5)
    fig.colorbar(pcm1, ax=ax1, label=r'$\log(|\lambda|)$')
    ax1.set_xlabel(r'$St_1$')
    ax1.set_ylabel(r'$St_2$')
    ax1.set_title('(a) Full Bispectrum')
    ax1.set_xlim(0, St1_max)
    ax1.set_ylim(St2_min, St2_max)

    # Plot (b): Zoomed-In Low-Frequency Region
    pcm2 = ax2.pcolormesh(ST1_zoom, ST2_zoom, log_lambda_zoom, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
    fig.colorbar(pcm2, ax=ax2, label=r'$\log(|\lambda|)$')
    ax2.set_xlabel(r'$St_1$')
    ax2.set_ylabel(r'$St_2$')
    ax2.set_title('(b) Low-Frequency Zoom')

    # Automatically select and mark the top M triads in the zoomed-in plot
    M = 6  # Number of triads to mark, adjustable if needed
    mask_zoom = (St1_vals >= 0) & (St1_vals <= zoom_St1_max) & (St2_vals >= zoom_St2_min) & (St2_vals <= zoom_St2_max)
    triads_zoom = triads_arr[mask_zoom]
    lambda_mags_zoom = lambda_mags[mask_zoom]
    St1_zoom_vals = St1_vals[mask_zoom]
    St2_zoom_vals = St2_vals[mask_zoom]

    if len(lambda_mags_zoom) > 0:
        top_idx = np.argsort(lambda_mags_zoom)[-M:][::-1]  # Indices of top M magnitudes
        top_triads = triads_zoom[top_idx]
        top_St1 = St1_zoom_vals[top_idx]
        top_St2 = St2_zoom_vals[top_idx]

        for (p1, p2, _), St1, St2 in zip(top_triads, top_St1, top_St2):
            ax2.plot(St1, St2, 'ro', markersize=8, markerfacecolor='none')
            ax2.text(St1, St2, f'({p1},{p2})', fontsize=8, ha='left', va='bottom')

    # Plot the fundamental line St_1 + St_2 = St_0 in the zoomed-in plot
    St1_line = np.linspace(0, zoom_St1_max, 100)
    St2_line = St0 - St1_line
    ax2.plot(St1_line, St2_line, 'k--', label=r'$St_1 + St_2 = St_0$')
    ax2.legend()

    # Set zoom plot limits

# Finalize and save the plot
plt.tight_layout()
output_file = os.path.join(analyzer.figures_dir, f"{analyzer.data_root}_BSMD_eig_St1St2_plane.png")
plt.savefig(output_file)
plt.close()
print(f"Plot saved to {output_file}")