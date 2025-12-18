# FFT Module for pyModal

High-performance FFT utilities with multi-backend support for cross-platform use (Linux Intel/AMD, macOS Apple Silicon).

## Benchmark Results Overview (Dec 2025)

### Test System

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core i7-14700 (20 cores, 28 threads) |
| **GPU** | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| **RAM** | 64 GB |
| **CUDA** | 12.0 |
| **Driver** | 580.82.09 |
| **Python** | 3.12.3 |
| **OS** | Pop!_OS Linux (Ubuntu-based) |

### Backends Tested

7 backends: `scipy`, `numpy`, `mkl`, `cupy`, `torch`, `torch_cuda`, `tensorflow`

### Key Finding: Power-of-2 vs Non-Power-of-2

**Rankings differ significantly based on FFT size type:**

| Scenario | Winner | Time @262K | Runner-up |
|----------|--------|------------|-----------|
| **Power-of-2** (e.g., 65536) | MKL | 0.46 ms | torch_cuda (2.3 ms) |
| **Non-power-of-2** (e.g., 80001) | torch_cuda | 1.57 ms | cupy (1.80 ms) |

### Non-Power-of-2 Slowdown Factor

GPU backends are **immune** to the non-power-of-2 penalty:

| Backend | Slowdown (non-pow2 vs pow2) |
|---------|------------------------------|
| cupy | 1.00x (no penalty) |
| torch_cuda | 1.01x (no penalty) |
| tensorflow | 1.07x (minimal) |
| torch | 1.53x |
| scipy | 2.87x |
| numpy | 3.41x |
| **MKL** | **12.73x** (severe) |

### Recommendation

| Data Type | Recommended Backend |
|-----------|---------------------|
| Power-of-2 sizes | **MKL** (5-20x faster than scipy) |
| Non-power-of-2 sizes | **CuPy** or **torch_cuda** (GPU immune to penalty) |
| Unknown/mixed sizes | **CuPy** (consistent performance) |
| No GPU available | **scipy** (most robust) |

### Performance at Real-World Sizes

| Size | Best Backend | Time (ms) |
|------|--------------|-----------|
| 80,001 | torch_cuda | 2.16 |
| 100,000 | torch_cuda | 2.06 |
| 150,000 | cupy | 3.37 |
| 200,000 | cupy | 4.52 |
| 262,144 (pow2) | MKL | 1.58 |

## Quick Start

```python
from fft import get_fft_func, get_available_backends, get_optimal_backend

# See what's available on your system
print(get_available_backends())  # e.g., ['scipy', 'numpy', 'mkl', 'cupy']

# Get optimal backend for your workload
backend = get_optimal_backend(array_size=65536, batch_size=64, gpu_resident=True)
fft_func = get_fft_func(backend)

# Compute FFT
import numpy as np
signal = np.random.randn(65536)
spectrum = fft_func(signal)
```

## Cross-Platform Support

| Platform | Recommended Backend | Notes |
|----------|---------------------|-------|
| **Linux + Intel CPU** | MKL | 2-10x faster than scipy |
| **Linux + AMD CPU** | scipy/numpy | MKL works but not optimized |
| **Linux + NVIDIA GPU** | CuPy (GPU-resident) | 5-75x faster for batches |
| **macOS Apple Silicon** | Accelerate | Uses vDSP framework |
| **macOS Intel** | MKL or scipy | MKL if available |

## Module Structure

| File | Description |
|------|-------------|
| `fft_backends.py` | Multi-backend FFT wrapper (scipy, numpy, MKL, CuPy, torch) |
| `gpu_utils.py` | GPU-accelerated batch FFT with memory management |
| `spectral_utils.py` | Spectral analysis (periodogram, Welch, Blackman-Tukey) |
| `complex_signal.py` | Test signal generator with harmonics and noise |
| `1_checks.py` | Validation tests (normalization, IFFT, Parseval) |
| `2_performance.py` | Performance benchmarks across backends |
| `3_interpolation.py` | Interpolation method comparison for non-uniform data |
| `4_methods.py` | Spectral estimation method comparison |
| `plots_for_Paul.py` | Custom analysis (requires external data file) |

## Verified Performance Results

### Single FFT Performance (Tested Dec 2025)

Intel CPU + RTX 4060 (8GB VRAM):

| Backend | 1K-64K FFT | Notes |
|---------|------------|-------|
| **MKL** | Fastest | 2-10x faster than scipy |
| **scipy** | Baseline | Always available |
| **numpy** | ~scipy | Similar performance |
| **CuPy (with transfer)** | Slower | PCIe overhead kills performance |

### Batch FFT Performance

**Key Finding**: Transfer overhead dominates. GPU only wins in **GPU-resident mode**.

| Config | MKL (ms) | GPU+Transfer (ms) | GPU-Resident (ms) | Resident Speedup |
|--------|----------|-------------------|-------------------|------------------|
| 4Kx64 | 3.87 | 0.59 | 0.02 | **235x** |
| 16Kx256 | 3.32 | 15.94 | 0.28 | **12x** |
| 64Kx256 | 16.13 | 65.42 | 2.24 | **7x** |
| 128Kx256 | 329.38 | 133.23 | 4.42 | **75x** |

**Conclusion**:
- **For single FFTs or data on CPU**: Use MKL
- **For pipelines where data stays on GPU**: Use CuPy GPU-resident mode (5-235x faster)

### Spectral Method Comparison (from 4_methods.py)

| Method | Error | Time | Best For |
|--------|-------|------|----------|
| **Blackman-Tukey** | 0.27% | 13.6ms | Lowest error |
| Periodogram | 1.73% | 21.9ms | Simple, fast |
| Welch | 1.98% | 15.0ms | Variance reduction |

### Interpolation Method Comparison (from 3_interpolation.py)

Best method varies by noise level - no single winner:
- **Low noise (1%)**: Zero-order
- **Medium noise (4%)**: Quintic Spline
- **Higher noise (7%)**: Linear
- **High noise (10%)**: Nearest

## Installation

### System Dependencies (Ubuntu/Debian/Pop!_OS)

```bash
# Required: Python 3.12+, HDF5, OpenBLAS
sudo apt install python3 python3-dev python3-venv \
    libhdf5-dev libopenblas-dev pkg-config

# Optional: NVIDIA GPU (driver only - CUDA comes via pip)
sudo apt install nvidia-driver-560  # or latest from Pop/Ubuntu repos
```

> **Note**: For best GPU performance, use the pip-installed CUDA runtime (`cupy-cuda12x` or `cupy-cuda13x`) rather than `nvidia-cuda-toolkit`. The pip wheels bundle the exact CUDA libraries needed.

### Quick Start (Pop!_OS with Intel oneAPI)

If you already have the environment set up, just run:

```bash
int25
```

This loads Intel oneAPI 2025 + the uv venv with all backends (cupy, mkl, torch, tensorflow).

### Python Environment Setup

```bash
# Create and activate virtual environment
uv venv ~/.venv
source ~/.venv/bin/activate
```

### Core Requirements (always needed)

```bash
uv pip install numpy scipy matplotlib h5py tabulate tqdm threadpoolctl
```

### All-in-One Install (copy-paste ready)

```bash
# System deps
sudo apt install -y python3 python3-dev python3-venv libhdf5-dev libopenblas-dev pkg-config

# Python packages (core)
uv pip install numpy scipy matplotlib h5py tabulate tqdm threadpoolctl

# Verify
python -c "import numpy, scipy, matplotlib, h5py, tabulate, tqdm; print('Core OK')"
```

### Intel MKL (Linux/macOS Intel - Recommended for CPU)

```bash
# Load Intel oneAPI environment (if available)
source /opt/intel/oneapi/setvars.sh  # or alias like 'int25'

# Install MKL FFT
uv pip install mkl-fft mkl-service \
    --index-url https://software.repos.intel.com/python/pypi \
    --extra-index-url https://pypi.org/simple

# Verify
python -c "import mkl_fft; print('MKL FFT OK')"
```

### NVIDIA GPU (Linux only)

```bash
# Check your driver's CUDA version
nvidia-smi | grep "CUDA Version"

# Install CuPy matching your driver's CUDA version:
uv pip install cupy-cuda12x   # For CUDA 12.x drivers (most common)
# OR
uv pip install cupy-cuda13x   # For CUDA 13.x drivers (newest)

# Verify
python -c "import cupy as cp; print(f'GPU: {cp.cuda.runtime.getDeviceProperties(0)[\"name\"].decode()}')"
python -c "import cupy as cp; a = cp.random.rand(10000); print(f'cuFFT OK: {cp.fft.fft(a).shape}')"
```

> **Tip**: CuPy wheels bundle their own CUDA runtime - no need for `nvidia-cuda-toolkit` apt package. Just match `cupy-cuda##x` to your driver's CUDA version from `nvidia-smi`.

### macOS Apple Silicon

The `accelerate` backend uses Apple's vDSP framework automatically. No extra installation needed, but it only supports power-of-2 FFT sizes.

### Summary Table

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| numpy | Array operations (2.x required) | `uv pip install numpy` |
| scipy | FFT baseline, signal processing | `uv pip install scipy` |
| matplotlib | Plotting | `uv pip install matplotlib` |
| h5py | HDF5 file I/O | `uv pip install h5py` |
| tabulate | Pretty tables | `uv pip install tabulate` |
| tqdm | Progress bars | `uv pip install tqdm` |
| threadpoolctl | Thread control | `uv pip install threadpoolctl` |
| mkl-fft | Intel MKL FFT (2-10x faster) | `uv pip install mkl-fft` |
| cupy-cuda12x | NVIDIA GPU (CUDA 12.x) | `uv pip install cupy-cuda12x` |
| cupy-cuda13x | NVIDIA GPU (CUDA 13.x) | `uv pip install cupy-cuda13x` |

### Recommended Versions (Dec 2025)

| Package | Min Version | Notes |
|---------|-------------|-------|
| Python | 3.12+ | 3.13 recommended |
| numpy | 2.0+ | Required for latest scipy |
| scipy | 1.14+ | Improved FFT performance |
| CuPy | 13.6+ | CUDA 13 + NumPy 2.3 support |

## Usage Examples

### Basic FFT with Optimal Backend

```python
from fft import get_fft_func, get_optimal_backend
import numpy as np

signal = np.random.randn(65536)

# Auto-select best backend for your platform
backend = get_optimal_backend(len(signal))
fft_func = get_fft_func(backend)
spectrum = fft_func(signal)
```

### GPU-Resident Batch FFT (Fastest for Pipelines)

```python
from fft import GPUBatchFFT
import numpy as np

processor = GPUBatchFFT()

# Transfer once, compute multiple times
signals = np.random.randn(256, 16384)

# GPU-resident mode - data stays on GPU
gpu_data = processor.to_gpu(signals)
gpu_spectra = processor.fft_gpu_resident(gpu_data, axis=1)
# ... more GPU operations (filtering, etc.) ...
gpu_filtered = processor.ifft_gpu_resident(gpu_spectra, axis=1)
result = processor.to_cpu(gpu_filtered)
```

### Spectral Analysis

```python
from fft import periodogram_rfft, welch_method, blackman_tukey_rfft, find_peaks
import numpy as np

# Generate test signal
fs = 1000  # Hz
t = np.arange(0, 1, 1/fs)
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + np.random.randn(len(t))*0.1

# Blackman-Tukey (lowest error)
freqs, psd = blackman_tukey_rfft(signal, fs)

# Find peaks
peak_freqs, peak_psd = find_peaks(freqs, psd, threshold=0.01)
print(f"Detected peaks at: {peak_freqs} Hz")
```

## Running Tests

### FFT Module Tests (in `fft/` folder)

```bash
cd fft/

# Validation tests (quick, ~10 seconds)
python 1_checks.py

# Performance benchmarks (slower, ~5 minutes)
python 2_performance.py

# Interpolation comparison (~30 seconds)
python 3_interpolation.py

# Spectral method comparison (~2 seconds)
python 4_methods.py
```

### Full pyModal Test Suite (from project root)

```bash
# Run comprehensive validation of POD, DMD, SPOD (22 tests)
python test_all.py

# This verifies:
# - FFT backend is correctly auto-detected
# - POD rank recovery, eigenvalues, orthonormality
# - DMD eigenvalue/frequency recovery
# - SPOD spectral analysis and tonal detection
# - Cross-method consistency
# - Heavy DOF tests (cylinder wake, Ginzburg-Landau, jet-like)
```

> **Tip**: Run `test_all.py` after installing new backends to verify they work correctly with the full analysis pipeline.

## Dependencies

**System (apt)**:
```bash
sudo apt install python3 python3-dev python3-venv libhdf5-dev libopenblas-dev pkg-config
```

**Required Python**:
```bash
uv pip install numpy scipy matplotlib h5py tabulate tqdm threadpoolctl
```

**Optional (high performance)**:
```bash
uv pip install mkl-fft mkl-service   # Intel MKL (2-10x faster on Intel CPUs)
uv pip install cupy-cuda12x          # NVIDIA GPU (CUDA 12.x)
uv pip install cupy-cuda13x          # NVIDIA GPU (CUDA 13.x - newest)
uv pip install torch                 # PyTorch FFT (ML pipelines)
```

## API Reference

### fft_backends.py

| Function | Description |
|----------|-------------|
| `get_fft_func(backend)` | Get FFT function for specified backend |
| `get_available_backends()` | List working backends on this system |
| `get_optimal_backend(size, batch, gpu_resident)` | Auto-select best backend |
| `benchmark_backends(size, iterations)` | Quick timing comparison |

### gpu_utils.py

| Class/Function | Description |
|----------------|-------------|
| `GPUBatchFFT` | Batch FFT processor with memory management |
| `should_use_gpu(size, batch, gpu_resident)` | Decision helper |
| `get_gpu_info()` | GPU memory and capability info |

### spectral_utils.py

| Function | Description |
|----------|-------------|
| `periodogram_rfft(x, fs)` | Standard periodogram PSD |
| `welch_method(x, fs)` | Welch's averaged periodogram |
| `blackman_tukey_rfft(x, fs)` | Blackman-Tukey (lowest error) |
| `find_peaks(freqs, psd)` | Peak detection |

## Automatic Backend Selection

pyModal **automatically detects** the best FFT backend at import time:

1. **Environment override**: `PYMODAL_FFT_BACKEND` env var (if set)
2. **MKL** (if available) - 2-10x faster than scipy on Intel CPUs
3. **scipy** (fallback) - always available

```python
# Check which backend is active
from configs import FFT_BACKEND
print(f"Using FFT backend: {FFT_BACKEND}")  # e.g., "mkl" or "scipy"
```

## Environment Variables

```bash
# Override auto-detected FFT backend
export PYMODAL_FFT_BACKEND=mkl   # Force MKL
export PYMODAL_FFT_BACKEND=cupy  # Force CuPy (GPU)

# Control MKL thread count
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

## Troubleshooting

### MKL not found (Linux)
```bash
# Quick: use int25 alias (Pop!_OS)
int25

# Manual: source Intel oneAPI
source /opt/intel/oneapi/setvars.sh
python -c "import mkl_fft; print('OK')"
```

### CuPy import fails
```bash
# Check driver is working
nvidia-smi                  # Should show GPU and CUDA version

# Check you installed the right CuPy version
nvidia-smi | grep "CUDA Version"  # e.g., "CUDA Version: 13.0"
# Install matching version: cupy-cuda12x for 12.x, cupy-cuda13x for 13.x

# If still failing, try reinstalling
uv pip uninstall cupy-cuda12x && uv pip install cupy-cuda12x
```

### Accelerate fails on macOS
- Only supports power-of-2 FFT sizes
- Pad your data to nearest power of 2

### GPU slower than expected
GPU is only faster than MKL when:
1. **Data stays on GPU** (`gpu_resident=True`)
2. Multiple operations in a pipeline

For single FFTs with transfer, **MKL always wins**.

## Normalization Convention

All backends use the standard **unnormalized** FFT:
```
X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
```
No scaling in forward FFT. IFFT divides by N.
