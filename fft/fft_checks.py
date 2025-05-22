"""
NOTE: All checks in this script assume the standard unnormalized FFT convention (no scaling by N).
This is the default in numpy, scipy, torch, tensorflow, pyfftw, and most scientific libraries:
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
No normalization is applied in the forward FFT; if you want a unitary FFT, divide by N (or sqrt(N)) as needed.

# IMPORTANT ON NORMALIZATION:
# Many FFT libraries (including numpy, scipy, torch, tensorflow, pyfftw) use the default 'unnormalized' (a.k.a. 'forward') convention:
#   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
#   x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2j*pi*k*n/N)
# That is, the forward FFT applies no scaling, and the inverse FFT divides by N.
# Some libraries (e.g., MATLAB's fft, or numpy/scipy with norm='ortho') allow or default to a 'unitary' (energy-preserving) normalization,
# where both the forward and inverse transform are scaled by 1/sqrt(N). This can make Parseval's theorem and other energy relations more symmetrical.
#
# **Mixing normalization conventions can lead to large differences in results!**
# For example, if you switch from an unnormalized FFT (no scaling) to a unitary FFT (scaling by 1/sqrt(N)),
# all your FFT amplitudes and spectral energies will change by factors of N or sqrt(N).
# This is a common source of confusion and bugs when changing FFT engines or porting code between libraries.
#
# Always check the normalization convention used by your FFT function, and apply the appropriate scaling if you need consistent results across engines.
# This script assumes the unnormalized convention for all checks and theoretical calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure project root is in sys.path so configs.py can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from fft_backends import get_fft_func, get_fft_backend_names

# ---- Configuration ----
FS = 1024
DURATION = 1.0
FREQ = 50
AMPLITUDE = 1.0
TOL = 1e-6


def generate_sine_wave(freq=FREQ, fs=FS, duration=DURATION, amplitude=AMPLITUDE, phase=0.0):
    t = np.arange(0, duration, 1/fs)
    x = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return t, x


def theoretical_fft_peak(amplitude, N):
    # Theory:
    # For a real-valued sine wave of amplitude A and length N, the DFT (FFT) of the signal will have two symmetric peaks:
    #   - At +f0 and -f0 (the positive and negative frequency bins corresponding to the sine frequency)
    #   - Each peak will have amplitude (A*N/2) in the unnormalized FFT (as returned by numpy/scipy/torch/etc)
    #   - This test checks whether the FFT backend returns the expected (unnormalized) amplitude, or if it applies a normalization (e.g., divided by N)
    #   - For rfft, the single-sided spectrum, the DC and Nyquist bins are not doubled, others are.
    return amplitude * N / 2


def test_fft_normalization(x, N, freq=FREQ, amplitude=AMPLITUDE, fs=FS):
    """
    Checks whether each FFT backend returns the expected (unnormalized) amplitude for a pure sine wave.
    Theory:
    - For a sine wave of amplitude A and length N, the FFT should have a peak of (A*N/2) at the sine frequency (for unnormalized FFTs).
    - This test confirms if the backend matches this convention, or applies a different normalization.
    """
    print(f"\nTesting FFT normalization for a {freq} Hz sine wave, {N} samples, amplitude={amplitude}\n")
    results = {}
    for backend in get_fft_backend_names():
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            # Compute frequency bins
            freqs = np.fft.fftfreq(N, 1/fs)
            # Find the bin closest to the test frequency
            idx = np.argmin(np.abs(freqs - freq))
            amp_measured = np.abs(X[idx])
            amp_theory = theoretical_fft_peak(amplitude, N)
            norm_ratio = amp_measured / amp_theory
            is_normalized = np.isclose(norm_ratio, 1.0, rtol=0.05)
            results[backend] = (amp_measured, amp_theory, norm_ratio, is_normalized)
            print(f"Backend: {backend:9s} | FFT peak: {amp_measured:.2f} | Theory: {amp_theory:.2f} | Ratio: {norm_ratio:.2f} | Normalized? {is_normalized}")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")
    # Optional: plot
    plt.figure(figsize=(8,4))
    for backend in get_fft_backend_names():
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            freqs = np.fft.fftfreq(N, 1/fs)
            plt.plot(freqs[:N//2], np.abs(X[:N//2]), label=backend)
        except Exception:
            continue
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('FFT Amplitude Spectrum by Backend')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fft_checks_spectrum.png')
    plt.close()
    print("Saved spectrum plot as fft_checks_spectrum.png")


def test_fft_inverse_consistency(x, N):
    """
    Checks that each backend's FFT and inverse FFT (IFFT) are mutually consistent.
    Theory:
    - The IFFT of the FFT of a signal should recover the original signal (within numerical precision).
    - This checks both the correctness of the implementation and the normalization convention (e.g., unnormalized FFTs are usually paired with IFFTs that divide by N).
    """
    print("\nTesting inverse FFT consistency for each backend\n")
    for backend in get_fft_backend_names():
        try:
            # Try to import the corresponding ifft
            if backend == 'scipy':
                from scipy.fft import ifft as ifft_func
            elif backend == 'numpy':
                from numpy.fft import ifft as ifft_func
            elif backend == 'pyfftw':
                import pyfftw
                def ifft_func(y):
                    return pyfftw.builders.ifft(y)()
            elif backend == 'tensorflow':
                import tensorflow as tf
                def ifft_func(y):
                    y_tf = tf.convert_to_tensor(y)
                    y_tf_complex = tf.cast(y_tf, tf.complex64)
                    return tf.signal.ifft(y_tf_complex).numpy().real
            elif backend == 'torch':
                import torch
                def ifft_func(y):
                    y_torch = torch.from_numpy(y)
                    y_torch_complex = y_torch.type(torch.complex64)
                    return torch.fft.ifft(y_torch_complex).numpy().real
            else:
                print(f"Backend: {backend:9s} | No IFFT implemented")
                continue
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            x_rec = ifft_func(X)
            # For numerical reasons, use np.allclose with a reasonable tolerance
            if np.allclose(x, x_rec.real, atol=TOL):
                print(f"Backend: {backend:9s} | IFFT consistency: PASS")
            else:
                print(f"Backend: {backend:9s} | IFFT consistency: FAIL (max abs diff: {np.max(np.abs(x - x_rec.real)):.2e})")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")


def test_fft_parseval(x, N):
    """
    Checks Parseval's theorem for the standard (complex, two-sided) FFT.
    Theory:
    - Parseval's theorem states that the total energy in the time domain equals the total energy in the frequency domain (after proper scaling):
        sum_n |x[n]|^2 = (1/N) * sum_k |X[k]|^2
      where X[k] is the (unnormalized) DFT of x[n].
    - This test verifies that each backend's FFT/IFFT pair is correctly normalized and energy-preserving.
    """
    print("\nTesting Parseval's theorem for each backend\n")
    E_time = np.sum(np.abs(x) ** 2)
    for backend in get_fft_backend_names():
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            E_freq = np.sum(np.abs(X) ** 2) / N
            if np.allclose(E_time, E_freq, rtol=TOL):
                print(f"Backend: {backend:9s} | Parseval: PASS | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f}")
            else:
                print(f"Backend: {backend:9s} | Parseval: FAIL | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f} | Diff: {abs(E_time-E_freq):.2e}")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")


def test_rfft_parseval(x, N):
    """
    Checks Parseval's theorem for the real FFT (rfft), i.e., the one-sided FFT for real-valued signals.
    Theory:
    - For a real signal x[n] of length N, the rfft returns only the non-negative frequency components.
    - Parseval's theorem for rfft:
        sum_n |x[n]|^2 = (1/N) * [|X[0]|^2 + |X[N/2]|^2 + 2*sum_{k=1}^{N/2-1} |X[k]|^2]   (for even N)
      where X[k] are the rfft bins, and the sum doubles the energy of all non-DC/non-Nyquist bins.
    - This test verifies that the rfft implementations preserve energy as expected.
    """
    print("\nTesting Parseval's theorem for real FFT (rfft)\n")
    E_time = np.sum(np.abs(x) ** 2)
    backends = ['numpy']
    try:
        import scipy.fft
        backends.append('scipy')
    except ImportError:
        pass
    try:
        import pyfftw
        backends.append('pyfftw')
    except ImportError:
        pass
    for backend in backends:
        try:
            if backend == 'numpy':
                rfft_func = np.fft.rfft
            elif backend == 'scipy':
                from scipy.fft import rfft as rfft_func
            elif backend == 'pyfftw':
                import pyfftw
                def rfft_func(y):
                    return pyfftw.builders.rfft(y)()
            else:
                continue
            Xr = rfft_func(x)
            # Parseval's theorem for rfft: sum(|x|^2) = (1/N) * (|X[0]|^2 + |X[N/2]|^2 + 2*sum_{k=1}^{N/2-1} |X[k]|^2)
            if N % 2 == 0:
                # Even N: Nyquist bin exists
                E_freq = (np.abs(Xr[0])**2 + np.abs(Xr[-1])**2 + 2*np.sum(np.abs(Xr[1:-1])**2)) / N
            else:
                # Odd N: No Nyquist bin
                E_freq = (np.abs(Xr[0])**2 + 2*np.sum(np.abs(Xr[1:])**2)) / N
            if np.allclose(E_time, E_freq, rtol=TOL):
                print(f"Backend: {backend:9s} | rfft Parseval: PASS | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f}")
            else:
                print(f"Backend: {backend:9s} | rfft Parseval: FAIL | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f} | Diff: {abs(E_time-E_freq):.2e}")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")


def main():
    t, x = generate_sine_wave()
    N = len(x)
    test_fft_normalization(x, N)
    test_fft_inverse_consistency(x, N)
    test_fft_parseval(x, N)
    test_rfft_parseval(x, N)

if __name__ == "__main__":
    main()
