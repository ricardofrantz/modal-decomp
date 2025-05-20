"""
Shared FFT backend selection and wrapper utilities for modal decomposition and benchmarking.
"""

from configs import FFT_BACKEND


def scipy_fft(x, axis=0):
    from scipy.fft import fft

    return fft(x, axis=axis)


def numpy_fft(x, axis=0):
    from numpy.fft import fft

    return fft(x, axis=axis)


def tensorflow_fft(x, axis=0):
    import tensorflow as tf

    x_tf = tf.convert_to_tensor(x)
    x_tf_complex = tf.cast(x_tf, tf.complex64)
    return tf.signal.fft(x_tf_complex).numpy()


def torch_fft(x, axis=0):
    import torch

    x_torch = torch.from_numpy(x)
    x_torch_complex = x_torch.type(torch.complex64)
    return torch.fft.fft(x_torch_complex, dim=axis).numpy()


def _pyfftw_fft_impl(x, axis=0):
    import numpy as np
    import pyfftw

    # Match dtype: if float64 or complex128, use complex128; else use complex64
    if np.issubdtype(x.dtype, np.floating):
        dtype = np.complex128 if x.dtype == np.float64 else np.complex64
        x = x.astype(dtype)
    elif x.dtype == np.complex128 or x.dtype == np.complex64:
        dtype = x.dtype
    else:
        dtype = np.complex64
        x = x.astype(dtype)
    a = pyfftw.empty_aligned(x.shape, dtype=dtype)
    a[:] = x
    fft_object = pyfftw.builders.fft(a, axis=axis)
    return fft_object()


# --- Intel MKL Placeholder ---
# Placeholder for Intel MKL.
# Note: MKL is often integrated directly into NumPy/SciPy builds from distributions
# like Anaconda or Intel's Python.
# Check your NumPy/SciPy configuration (np.show_config()) to see if MKL is used.
# If MKL is part of your NumPy/SciPy, then the 'numpy' or 'scipy' backends
# are already effectively using MKL-optimized FFTs.
#
# def mkl_fft(x, axis=0):
#     # This would typically require a specific library or a way to call MKL FFTs
#     # directly, separate from how NumPy/SciPy use MKL internally. This is uncommon.
#     # For example, if using a hypothetical 'pyMKLfft' library:
#     # import pyMKLfft
#     # return pyMKLfft.fft(x, axis=axis)
#     raise NotImplementedError("Direct MKL backend not implemented. Check NumPy/SciPy for MKL integration.")

FFT_BACKENDS = {
    "scipy": scipy_fft,
    "numpy": numpy_fft,
    "tensorflow": tensorflow_fft,
    "torch": torch_fft,
    # "mkl": mkl_fft, # Uncomment if you establish a direct MKL FFT binding separate from NumPy/SciPy
    # Add more here as needed!
}

try:
    import pyfftw  # Attempt to import to check availability

    FFT_BACKENDS["pyfftw"] = _pyfftw_fft_impl
    # print("PyFFTW backend enabled.")
except ImportError:
    print("PyFFTW not installed or found. PyFFTW backend will be unavailable.")
except Exception as e:
    # Catch any other error during PyFFTW probing/loading
    print(f"Error loading PyFFTW backend: {e}. PyFFTW backend will be unavailable.")


def get_fft_func(backend=None):
    backend = backend or FFT_BACKEND
    if backend not in FFT_BACKENDS:
        raise ValueError(f"Unknown FFT_BACKEND: {backend}")
    return FFT_BACKENDS[backend]


def get_fft_backend_names():
    return list(FFT_BACKENDS.keys())
