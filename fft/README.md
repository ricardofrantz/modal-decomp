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

### Core (always works)
```bash
uv pip install scipy numpy
```

### Intel MKL (Linux/macOS Intel - Recommended for CPU)

```bash
# Load Intel oneAPI environment (if available)
source /opt/intel/oneapi/setvars.sh  # or alias like 'int25'

# Install via uv
uv pip install mkl_fft mkl-service \
    --index-url https://software.repos.intel.com/python/pypi \
    --extra-index-url https://pypi.org/simple

# Verify
python -c "import mkl_fft; print('MKL FFT OK')"
```

### NVIDIA GPU (Linux only)

```bash
# Prerequisites: NVIDIA driver + CUDA toolkit
nvidia-smi                           # Check driver
sudo apt install nvidia-cuda-toolkit  # Ubuntu/Debian
nvcc --version                       # Verify CUDA

# Install CuPy for CUDA 12.x
uv pip install cupy-cuda12x

# Verify
python -c "import cupy as cp; print(f'GPU OK: {cp.cuda.Device(0).compute_capability}')"
```

### macOS Apple Silicon

The `accelerate` backend uses Apple's vDSP framework automatically. No extra installation needed, but it only supports power-of-2 FFT sizes.

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

## Dependencies

Required:
- `numpy`
- `scipy`
- `matplotlib`
- `tabulate`

Optional (for high performance):
- `mkl_fft` - Intel MKL FFT (Linux/macOS Intel)
- `cupy-cuda12x` - NVIDIA GPU FFT (Linux only)
- `torch` - PyTorch FFT (if using ML pipelines)

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

## Environment Variables

```bash
# Override default FFT backend
export PYMODAL_FFT_BACKEND=mkl

# Control MKL thread count
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

## Troubleshooting

### MKL not found (Linux)
```bash
source /opt/intel/oneapi/setvars.sh
python -c "import mkl_fft; print('OK')"
```

### CuPy import fails
```bash
nvcc --version              # Check CUDA toolkit
ldconfig -p | grep cufft    # Check libcufft
sudo apt install nvidia-cuda-toolkit  # If missing
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
