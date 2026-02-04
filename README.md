```
██████╗ ██╗   ██╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗
██╔══██╗╚██╗ ██╔╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║
██████╔╝ ╚████╔╝ ██╔████╔██║██║   ██║██║  ██║███████║██║
██╔═══╝   ╚██╔╝  ██║╚██╔╝██║██║   ██║██║  ██║██╔══██║██║
██║        ██║   ██║ ╚═╝ ██║╚██████╔╝██████╔╝██║  ██║███████╗
╚═╝        ╚═╝   ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
```

[![CI](https://github.com/ricardofrantz/pyModal/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardofrantz/pyModal/actions/workflows/ci.yml)
![Python 3.14](https://img.shields.io/badge/Python-3.14-blue)
![macOS](https://img.shields.io/badge/macOS-26-lightgrey)
![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-orange)

# pyModal — modal decompositions in pure Python

A lightweight, zero-MPI toolkit for extracting coherent structures from spatiotemporal data. Every algorithm fits in a few hundred readable lines—study, tweak, or extend the maths without fighting a framework.

**Methods implemented:** POD, DMD, SPOD, BSMD (with ST-POD planned)

## Quick Start

```bash
git clone https://github.com/ricardofrantz/pyModal.git
cd pyModal
pip install numpy scipy matplotlib h5py tqdm

# Run the 2D benchmark examples
python examples_2d.py
```

This generates synthetic flow fields (Double Gyre, Taylor-Green Vortex, Cylinder Wake) and runs POD, DMD, and SPOD on each—figures saved to `./figs_examples/`.

---

## Usage

### Python API

```python
from pod import PODAnalyzer
from spod import SPODAnalyzer
from dmd import DMDAnalyzer
from examples_2d import cylinder_wake, make_loader

# Generate synthetic data (or load your own)
data = cylinder_wake(Nx=100, Ny=50, Nt=500, Re=100)
loader = make_loader(data)

# --- POD ---
pod = PODAnalyzer(
    file_path="cylinder",
    data_loader=loader,
    n_modes_save=10,
)
pod.run_analysis()
print(f"POD modes: {pod.modes.shape}")           # (Nspace, n_modes)
print(f"Eigenvalues: {pod.eigenvalues[:3]}")     # Energy per mode

# --- DMD ---
dmd = DMDAnalyzer(
    file_path="cylinder",
    data_loader=loader,
    n_modes_save=10,
)
dmd.load_and_preprocess()
dmd.perform_dmd()
print(f"DMD eigenvalues: {dmd.eigenvalues[:3]}")  # Complex, |λ|<1 = decay

# --- SPOD ---
spod = SPODAnalyzer(
    file_path="cylinder",
    data_loader=loader,
    nfft=128,
    overlap=0.5,
)
spod.run()
spod.perform_spod()
print(f"SPOD frequencies: {spod.freq[:5]} Hz")
```

### Loading Your Own Data

pyModal auto-detects `.mat`, `.h5`, and `.npz` files. Your data should have:

```python
# Required structure
{
    'q': np.ndarray,  # shape (Ns, Nspace) — snapshots × flattened spatial points
    'dt': float,      # time step between snapshots
    'Nx': int,        # grid points in x
    'Ny': int,        # grid points in y
    'x': np.ndarray,  # x-coordinates (optional)
    'y': np.ndarray,  # y-coordinates (optional)
}
```

Example with a `.mat` file:

```python
from pod import PODAnalyzer

pod = PODAnalyzer(file_path="./data/my_simulation.mat", n_modes_save=20)
pod.run_analysis()
```

### Command Line

Each script supports staged execution:

```bash
# Full analysis (prep → compute → plot)
python pod.py --data ./data/my_file.mat

# Staged execution
python spod.py --prep      # preprocess only
python spod.py --compute   # compute decomposition
python spod.py --plot      # generate figures

# Run all methods sequentially
python pyModal.py --data ./data/my_file.mat

# Run specific method
python pyModal.py --spod --data ./data/my_file.mat
```

---

## Methods

| Method | Use Case | Key Output |
|--------|----------|------------|
| **POD** | Energy-ranked spatial modes | `modes`, `eigenvalues`, `time_coefficients` |
| **DMD** | Growth rates, oscillatory structures | `modes`, `eigenvalues` (complex), `frequencies` |
| **SPOD** | Frequency-resolved coherent structures | `modes[freq]`, `eigenvalues[freq]` |
| **BSMD** | Triadic interactions, nonlinear coupling | Bispectral modes at frequency triads |

### POD — Proper Orthogonal Decomposition

Energy-optimal spatial modes via SVD of mean-subtracted snapshots.

```python
pod = PODAnalyzer(file_path="data.mat", n_modes_save=10)
pod.run_analysis()

# Results
pod.modes            # (Nspace, n_modes) — spatial modes
pod.eigenvalues      # (n_modes,) — energy per mode
pod.time_coefficients  # (Ns, n_modes) — temporal evolution
```

### DMD — Dynamic Mode Decomposition

Extracts eigenvalues/modes of the best-fit linear operator.

```python
dmd = DMDAnalyzer(file_path="data.mat", n_modes_save=10)
dmd.load_and_preprocess()
dmd.perform_dmd()

# Eigenvalues on complex plane: |λ| < 1 = decaying, |λ| > 1 = growing
frequencies = np.angle(dmd.eigenvalues) / (2 * np.pi * dmd.dt)
```

### SPOD — Spectral POD

Frequency-resolved modes under stationary assumptions. [Towne, Schmidt & Colonius (2018)](https://arxiv.org/abs/1708.04393)

```python
spod = SPODAnalyzer(file_path="data.mat", nfft=256, overlap=0.5)
spod.run()
spod.perform_spod()

# Modes at each frequency
spod.freq              # frequency bins (Hz)
spod.eigenvalues[f]    # energy at frequency index f
spod.modes[f]          # spatial modes at frequency f
```

### BSMD — Bispectral Mode Decomposition

Third-order interactions revealing nonlinear energy transfer. [Nekkanti et al. (2025)](https://arxiv.org/abs/2502.15091)

```bash
# BSMD reuses cached FFT blocks from SPOD
python spod.py --data ./data/my_file.mat   # creates results_spod/
python bmsd.py --data ./data/my_file.mat   # reuses cache, outputs to results_bsmd/
```

---

## Configuration

Edit `configs.py` or pass a JSON/YAML file:

```python
from configs import load_config
load_config("my_settings.yaml")
```

Key settings:

```python
FFT_BACKEND = "scipy"   # or "mkl", "numpy", "accelerate" (macOS)
FIG_DPI = 500
WINDOW_TYPE = "hamming"
```

Override FFT backend via environment variable:

```bash
PYMODAL_FFT_BACKEND=mkl python spod.py
```

---

## Installation

```bash
pip install numpy scipy matplotlib h5py tqdm
```

**Performance tips:**

```bash
# Intel MKL (2-10x faster FFTs on Intel CPUs)
conda install mkl_fft

# Apple Silicon
pip install pyobjc-framework-Accelerate
```

Check active optimizations:

```bash
python -m parallel_utils
```

---

## Project Structure

```
pyModal/
├── src/pymodal/           # Main package (pip installable)
│   ├── pod.py             # Proper Orthogonal Decomposition
│   ├── dmd.py             # Dynamic Mode Decomposition
│   ├── spod.py            # Spectral POD
│   ├── bmsd.py            # Bispectral Mode Decomposition
│   ├── cli.py             # Command-line interface
│   ├── core/              # Shared utilities
│   │   ├── base.py        # BaseAnalyzer class
│   │   ├── config.py      # Global settings
│   │   ├── io.py          # Data loaders
│   │   └── parallel.py    # Parallelization
│   └── fft/               # FFT backends
├── examples/              # Benchmark examples
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── pyproject.toml         # Package configuration
```

---

## References

- **SPOD:** Towne, Schmidt & Colonius (2018) — [arXiv:1708.04393](https://arxiv.org/abs/1708.04393)
- **BSMD:** Nekkanti, Pickering, Schmidt & Colonius (2025) — [arXiv:2502.15091](https://arxiv.org/abs/2502.15091)
- **ST-POD:** Yeung & Schmidt (2025) — [arXiv:2502.09746](https://arxiv.org/abs/2502.09746) *(planned)*

---

## Developer Notes

<details>
<summary>Click to expand mathematical details and extension guide</summary>

### Mathematical Overview

**POD** performs a weighted SVD of mean-subtracted snapshots. Depending on dimensions, it solves either the temporal or spatial covariance problem.

**SPOD** solves an eigenvalue problem for the cross-spectral density matrix. FFT blocks are computed with Welch's method. For each frequency bin, the weighted matrix M = X^H W X is diagonalized.

**BSMD** analyzes triadic interactions. For a triad (p1, p2, p3) with f_p1 + f_p2 = f_p3, it forms matrices A and B from cached FFT blocks and solves C = A† W B, C a = λ a.

### Caching

FFT blocks (`qhat`) are stored in HDF5 files. Both SPOD and BSMD check for existing caches before recomputing—BSMD can reuse SPOD caches directly.

### Extending

- Subclass `BaseAnalyzer` for new decompositions
- Results go in `results_*/`, figures in `figs_*/`
- Override settings via `configs.load_config("custom.yaml")`

</details>

