"""
FFT Performance Benchmarking Suite for pyModal.

=============================================================================
WHAT THIS SCRIPT MEASURES
=============================================================================

1. Single FFT Performance:
   - Tests all available backends (scipy, numpy, mkl, cupy, torch, etc.)
   - Tests various signal sizes from 1K to 128K samples
   - Tests both power-of-2 and non-power-of-2 sizes
   - For single FFTs, MKL typically wins on CPU

2. Batch FFT Performance (GPU comparison):
   - Tests batches of 1, 16, 64, 128 FFTs at once
   - GPU (CuPy) excels at batch processing in GPU-RESIDENT mode
   - Expected results (RTX 4060):
     * With transfer: MKL typically wins (PCIe overhead ~0.5ms)
     * GPU-resident (no transfer): GPU wins 12-100x for large batches
     * Use GPUBatchFFT class for pipelines where data stays on GPU

=============================================================================
RUNNING THE BENCHMARK
=============================================================================

    # Full benchmark (takes 5-10 minutes)
    python 2_performance.py

    # Quick benchmark only (from Python)
    from pymodal.fft.fft_backends import benchmark_backends
    results = benchmark_backends(size=65536, iterations=50)

    # Batch benchmark only
    from pymodal.fft.gpu_utils import benchmark_cpu_vs_gpu
    results = benchmark_cpu_vs_gpu()

=============================================================================
INTERPRETING RESULTS
=============================================================================

Backend Selection Guide:
    - MKL:  Best for large single FFTs on CPU (2-10x faster than scipy)
    - CuPy: Best for batches >= 64 or single FFTs >= 64K points
    - scipy: Good default, always available

Performance Notes:
    - Non-power-of-2 sizes are slower due to less efficient algorithms
    - GPU transfer overhead is ~0.5ms per operation
    - GPU plan caching helps repeated same-size FFTs

=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from fft_backends import get_fft_func, get_fft_backend_names, get_available_backends
import timeit
import json
import matplotlib.pyplot as plt
from scipy import signal

DPI=500

# Function to generate 1D signals with realistic noise
def generate_signal(size):
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Create time vector
    t = np.linspace(0, 1, size)
    
    # Generate a signal with multiple frequency components
    # More realistic signal with 3 frequency components
    signal_clean = (
        0.5 * np.sin(2 * np.pi * 10 * t) +  # Low frequency component
        0.3 * np.sin(2 * np.pi * 25 * t) +  # Mid frequency component
        0.2 * np.sin(2 * np.pi * 50 * t)    # High frequency component
    )
    
    # Add colored noise (more realistic than white noise)
    # Pink noise (1/f noise) is common in natural signals
    noise_level = 0.05
    white_noise = np.random.normal(0, noise_level, size)
    
    # Create pink noise by filtering white noise
    # Use a simple lowpass filter to create colored noise
    colored_noise = signal.lfilter([1.0], [1.0, -0.9], white_noise)
    
    # Add some impulsive noise (outliers) to simulate measurement errors
    impulse_locations = np.random.choice(size, size=int(size * 0.01), replace=False)
    impulse_noise = np.zeros(size)
    impulse_noise[impulse_locations] = np.random.normal(0, noise_level * 5, size=len(impulse_locations))
    
    # Combine signal and noise
    noisy_signal = signal_clean + colored_noise + impulse_noise
    
    return noisy_signal.astype(np.float32)

# Function to safely compute FFT and handle errors
def safe_fft(signal, fft_func, *args, **kwargs):
    try:
        return fft_func(signal, *args, **kwargs)
    except Exception as e:
        print(f"Error computing FFT: {e}")
        return None

# Function to run the FFT for all libraries
def compare_fft(size, N_times=3, discard=1):
    """
    Benchmark all available FFT backends for a given signal size.
    Args:
        size (int): Signal size.
        N_times (int): Number of times to repeat the timing and average (after discarding warmup runs). Default 3.
        discard (int): Number of initial timing runs to discard (warmup). Default 1.
    Returns:
        dict: Timings and errors for each backend, plus the reference backend.
    """
    sig = generate_signal(size)
    backend_names = get_fft_backend_names()
    result_dict = {}
    timings = {}
    results = {}
    # Benchmark all backends
    for backend in backend_names:
        fft_func = get_fft_func(backend)
        try:
            times = []
            total_runs = N_times + discard
            for i in range(total_runs):
                def wrapper():
                    return fft_func(sig)
                t = timeit.timeit(wrapper, number=10)
                if i >= discard:
                    times.append(t)
            avg_time = float(np.mean(times)) if times else 0
            res = np.abs(wrapper())
            timings[backend] = avg_time
            results[backend] = res
        except Exception as e:
            print(f"Error with {backend} FFT: {e}")
            timings[backend] = None
            results[backend] = None
    # Choose the fastest valid backend as reference
    valid = {b: timings[b] for b in backend_names if timings[b] is not None}
    if not valid:
        for b in backend_names:
            result_dict[f"{b}_fft_time"] = 0
            result_dict[f"{b}_error"] = 0
        return result_dict
    ref_backend = min(valid, key=valid.get)
    ref_result = results[ref_backend]
    for b in backend_names:
        result_dict[f"{b}_fft_time"] = timings[b] if timings[b] is not None else 0
        if results[b] is not None:
            result_dict[f"{b}_error"] = float(np.mean(np.abs(ref_result[:len(results[b])] - results[b][:len(ref_result)])))
        else:
            result_dict[f"{b}_error"] = 0
    result_dict['reference_backend'] = ref_backend
    return result_dict

# Test with different sizes for 1D signals, including non-powers of 2
# Generate powers of two and nearby off values, up to 256K (262144)
powers = list(range(10, 19))  # 2^10 (1024) to 2^18 (262144)
sizes_pow2 = [2 ** p for p in powers]
sizes_off = []
for n in sizes_pow2:
    # Add both +1 and -1, +3 and -3 neighbors, but only if positive, not a power of two, and <= 262144
    for delta in [-3, -1, +1, +3]:
        off_val = n + delta
        if 0 < off_val <= 262144 and (off_val & (off_val - 1)) != 0:
            sizes_off.append(off_val)

# Add real-world sizes from JFM_CS project (80001 samples is typical)
real_world_sizes = [80001, 100000, 150000, 200000]
sizes_off.extend(real_world_sizes)

# Combine, deduplicate, and sort
sizes = sorted(set(sizes_pow2 + sizes_off))

print("Testing", len(sizes), "sizes:", sizes)

backend_names = get_fft_backend_names()
N_times = 3  # Default number of repetitions for timing

# Initialize results dictionary dynamically for all available backends
results = {}
for backend in backend_names:
    results[f"{backend}_fft_time"] = []
    results[f"{backend}_error"] = []
results['reference_backend'] = []

for size in sizes:
    print(f"Processing size: {size}")
    result = compare_fft(size, N_times=N_times)
    for key in result:
        if key in results:
            results[key].append(result[key])
    results['reference_backend'].append(result.get('reference_backend', ''))
    print(f"Completed size: {size}")

# Save the results to a JSON file
with open('2_performance.json', 'w') as f:
    results_copy = {k: [float(x) if isinstance(x, (int, float)) else x for x in v] for k, v in results.items()}
    results_copy['sizes'] = [float(x) for x in sizes]  # Convert sizes to float
    json.dump(results_copy, f, indent=4)
    print("Results saved to 2_performance.json")

# Function to plot the results from the JSON file
def plot_fft_results(json_file='2_performance.json'):
    # Load the results from the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    sizes = np.array(data['sizes'])
    sizes_int = [int(s) for s in sizes]
    backend_names = [k[:-9] for k in data.keys() if k.endswith('_fft_time')]

    # Separate power-of-2 and non-power-of-2 indices
    pow2_idx = [i for i, s in enumerate(sizes_int) if (s & (s-1)) == 0]
    non_pow2_idx = [i for i, s in enumerate(sizes_int) if (s & (s-1)) != 0]

    pow2_sizes = sizes[pow2_idx]
    non_pow2_sizes = sizes[non_pow2_idx]

    # Helper function to plot one category
    def plot_category(cat_sizes, cat_idx, title_suffix, filename_suffix):
        if len(cat_idx) == 0:
            return {}
        plt.figure(figsize=(12, 8))
        fit_results = {}
        for backend in backend_names:
            times = np.array(data[f"{backend}_fft_time"])
            cat_times = times[cat_idx]
            line_plot, = plt.plot(cat_sizes, cat_times, marker='o', label=f"{backend} FFT")
            # Only fit where times > 0 (exclude failed runs)
            valid = cat_times > 0
            if np.sum(valid) > 1:
                log_sizes = np.log(cat_sizes[valid])
                log_times = np.log(cat_times[valid])
                slope, intercept = np.polyfit(log_sizes, log_times, 1)
                fit_results[backend] = (slope, intercept)
                fit_line = np.exp(intercept) * cat_sizes**slope
                plt.plot(cat_sizes, fit_line, linestyle='--',
                        label=f"{backend} fit (slope={slope:.2f})", color=line_plot.get_color())
                # Annotate slope at the largest size
                x_annot = cat_sizes[-1]
                y_annot = np.exp(intercept) * x_annot**slope
                plt.annotate(f"slope={slope:.2f}",
                             xy=(x_annot, y_annot),
                             xytext=(10, 0),
                             textcoords='offset points',
                             color=line_plot.get_color(),
                             fontsize=10,
                             va='center',
                             fontweight='bold')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Signal Size (samples)')
        plt.ylabel('Time (seconds) for 10 iterations')
        plt.title(f'FFT Performance Comparison {title_suffix}')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig(f'2_performance_{filename_suffix}.png', dpi=DPI, bbox_inches='tight')
        plt.close()
        return fit_results

    # Plot power-of-2 sizes
    print("\n=== POWER-OF-2 SIZES ===")
    print(f"Sizes: {[int(s) for s in pow2_sizes]}")
    fit_pow2 = plot_category(pow2_sizes, pow2_idx,
                              '(Power-of-2 Sizes)', 'pow2')

    # Plot non-power-of-2 sizes
    print("\n=== NON-POWER-OF-2 SIZES ===")
    print(f"Sizes: {[int(s) for s in non_pow2_sizes[:8]]}...")
    fit_non_pow2 = plot_category(non_pow2_sizes, non_pow2_idx,
                                  '(Non-Power-of-2 Sizes)', 'non_pow2')

    # Print ranking comparison
    print("\n=== RANKING COMPARISON ===")
    pow2_avgs = {}
    non_pow2_avgs = {}
    for backend in backend_names:
        times = np.array(data[f"{backend}_fft_time"])
        pow2_times = times[pow2_idx]
        non_pow2_times = times[non_pow2_idx]
        valid_pow2 = pow2_times > 0
        valid_non_pow2 = non_pow2_times > 0
        if np.any(valid_pow2):
            pow2_avgs[backend] = np.mean(pow2_times[valid_pow2])
        if np.any(valid_non_pow2):
            non_pow2_avgs[backend] = np.mean(non_pow2_times[valid_non_pow2])

    pow2_rank = sorted(pow2_avgs.keys(), key=lambda x: pow2_avgs[x])
    non_pow2_rank = sorted(non_pow2_avgs.keys(), key=lambda x: non_pow2_avgs[x])

    print(f"Power-of-2 ranking:     {' > '.join(pow2_rank)}")
    print(f"Non-power-of-2 ranking: {' > '.join(non_pow2_rank)}")

    print("\n=== SLOWDOWN FACTOR (non-pow2 / pow2) ===")
    for backend in backend_names:
        if backend in pow2_avgs and backend in non_pow2_avgs:
            slowdown = non_pow2_avgs[backend] / pow2_avgs[backend]
            print(f"{backend:12s}: {slowdown:.2f}x slower for non-power-of-2")

    print("\n=== AVERAGE TIMINGS ===")
    print(f"{'Backend':12s} {'Pow2 (ms)':>12s} {'Non-Pow2 (ms)':>14s}")
    print("-" * 40)
    for backend in backend_names:
        p2 = pow2_avgs.get(backend, 0) * 1000
        np2 = non_pow2_avgs.get(backend, 0) * 1000
        print(f"{backend:12s} {p2:12.3f} {np2:14.3f}")

    fastest_pow2 = pow2_rank[0] if pow2_rank else "N/A"
    fastest_non_pow2 = non_pow2_rank[0] if non_pow2_rank else "N/A"
    print(f"\nFastest for power-of-2:     {fastest_pow2}")
    print(f"Fastest for non-power-of-2: {fastest_non_pow2}")

    # Return fit results for both
    fit_results = {'pow2': fit_pow2, 'non_pow2': fit_non_pow2}

    # Print which backend is overall fastest (lowest average timing)
    avg_timings = {}
    for backend in backend_names:
        times = np.array(data[f"{backend}_fft_time"])
        valid = times > 0
        if np.any(valid):
            avg_timings[backend] = np.mean(times[valid])
        else:
            avg_timings[backend] = float('inf')
    fastest_backend = min(avg_timings, key=avg_timings.get)
    print("\n--- Linear Fit Results (log-log space) ---")
    print("Power-of-2 sizes:")
    for backend in backend_names:
        if backend in fit_results.get('pow2', {}):
            slope, intercept = fit_results['pow2'][backend]
            print(f"  {backend}: slope={slope:.3f}, intercept={intercept:.3f}")
    print("Non-power-of-2 sizes:")
    for backend in backend_names:
        if backend in fit_results.get('non_pow2', {}):
            slope, intercept = fit_results['non_pow2'][backend]
            print(f"  {backend}: slope={slope:.3f}, intercept={intercept:.3f}")
    print(f"\nOverall fastest backend (lowest mean timing): {fastest_backend} (mean time: {avg_timings[fastest_backend]:.6g} s for 10 runs)")

    # Plot errors
    plt.figure(figsize=(12, 8))
    for backend in backend_names:
        plt.plot(sizes, data[f"{backend}_error"], marker='o', label=f"{backend} Error")
    for size in sizes:
        size_int = int(size)
        if size_int & (size_int - 1) != 0:
            plt.axvline(x=size, color='lightgray', linestyle='--', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Signal Size (samples)')
    plt.ylabel('Mean Absolute Error (vs Reference)')
    plt.title('FFT Accuracy Comparison Across Libraries (1D Signals)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig('2_performance_accuracy.png', dpi=DPI, bbox_inches='tight')
    # Sample signal and FFT
    plt.figure(figsize=(12, 8))
    sample_size = 1000
    sample_signal = generate_signal(sample_size)
    t = np.linspace(0, 1, sample_size)
    plt.subplot(2, 1, 1)
    plt.plot(t, sample_signal)
    plt.title('Sample 1D Signal with Realistic Noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 1, 2)
    try:
        # Use the reference backend for FFT
        ref_backend = data['reference_backend'][0] if 'reference_backend' in data and data['reference_backend'] else backend_names[0]
        fft_func = get_fft_func(ref_backend)
        fft_complex = fft_func(sample_signal)
        fft_magnitude = np.abs(fft_complex)
        dt = t[1] - t[0]
        freqs = np.fft.fftfreq(len(sample_signal), dt)
        positive_freq_idx = len(freqs) // 2
        plt.plot(freqs[:positive_freq_idx], fft_magnitude[:positive_freq_idx])
        peak_freqs = [10, 25, 50]
        for freq in peak_freqs:
            idx = np.argmin(np.abs(freqs[:positive_freq_idx] - freq))
            plt.plot(freqs[idx], fft_magnitude[idx], 'ro')
            plt.text(freqs[idx], fft_magnitude[idx], f"{freq} Hz", verticalalignment='bottom', horizontalalignment='center')
    except Exception as e:
        print(f"Error plotting FFT: {e}")
        plt.text(0.5, 0.5, f"Error computing FFT: {e}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('FFT Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('2_performance_signal.png', dpi=DPI, bbox_inches='tight')
    print("Plots saved: 2_performance_pow2.png, 2_performance_non_pow2.png, 2_performance_accuracy.png, 2_performance_signal.png")


# =============================================================================
# Batch Benchmark (CPU vs GPU comparison)
# =============================================================================

def benchmark_batch_fft(sizes=None, batch_sizes=None, iterations=10):
    """
    Compare CPU (scipy/mkl) vs GPU (cupy) for batch FFT operations.

    This benchmark shows where GPU excels:
    - Large batches (64+) of any size FFT
    - Single large FFTs (64K+)

    Args:
        sizes: List of FFT sizes. Default: [4096, 16384, 65536]
        batch_sizes: List of batch sizes. Default: [1, 16, 64, 128]
        iterations: Timing iterations per test

    Returns:
        dict with timing results and speedup factors
    """
    import time

    if sizes is None:
        sizes = [4096, 16384, 65536]
    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 128]

    # Get available backends
    available = get_available_backends()
    use_mkl = 'mkl' in available
    use_gpu = 'cupy' in available

    results = {}

    # Import backends
    from scipy.fft import fft as scipy_fft
    if use_mkl:
        from mkl_fft import fft as mkl_fft_func
    if use_gpu:
        import cupy as cp

    print("\n" + "=" * 60)
    print("BATCH FFT BENCHMARK (CPU vs GPU)")
    print("=" * 60)
    print(f"{'Config':>15s} {'scipy (ms)':>12s}", end="")
    if use_mkl:
        print(f" {'MKL (ms)':>12s}", end="")
    if use_gpu:
        print(f" {'GPU (ms)':>12s} {'GPU Speedup':>12s}", end="")
    print()
    print("-" * 70)

    for size in sizes:
        for batch in batch_sizes:
            key = f'{size}x{batch}'
            # Generate batch of complex signals
            data = np.random.randn(batch, size).astype(np.float32) + \
                   1j * np.random.randn(batch, size).astype(np.float32)

            # scipy timing (baseline)
            start = time.perf_counter()
            for _ in range(iterations):
                scipy_fft(data, axis=1)
            scipy_time = (time.perf_counter() - start) / iterations * 1000
            results[key] = {'scipy_ms': round(scipy_time, 3)}

            # MKL timing
            if use_mkl:
                start = time.perf_counter()
                for _ in range(iterations):
                    mkl_fft_func(data, axis=1)
                mkl_time = (time.perf_counter() - start) / iterations * 1000
                results[key]['mkl_ms'] = round(mkl_time, 3)
                results[key]['mkl_speedup'] = round(scipy_time / mkl_time, 2)

            # GPU timing
            if use_gpu:
                # Warmup
                try:
                    gpu_data = cp.asarray(data)
                    _ = cp.fft.fft(gpu_data, axis=1)
                    cp.cuda.Stream.null.synchronize()

                    start = time.perf_counter()
                    for _ in range(iterations):
                        gpu_data = cp.asarray(data)
                        result = cp.fft.fft(gpu_data, axis=1)
                        _ = cp.asnumpy(result)
                        cp.cuda.Stream.null.synchronize()
                    gpu_time = (time.perf_counter() - start) / iterations * 1000

                    results[key]['gpu_ms'] = round(gpu_time, 3)
                    # Use best CPU time for speedup
                    best_cpu = scipy_time
                    if use_mkl:
                        best_cpu = min(scipy_time, mkl_time)
                    results[key]['gpu_speedup'] = round(best_cpu / gpu_time, 2)
                except Exception as e:
                    results[key]['gpu_ms'] = 'N/A'
                    results[key]['gpu_speedup'] = 'N/A'

            # Print row
            print(f"{key:>15s} {scipy_time:>12.3f}", end="")
            if use_mkl:
                print(f" {results[key].get('mkl_ms', 'N/A'):>12}", end="")
            if use_gpu:
                print(f" {results[key].get('gpu_ms', 'N/A'):>12}", end="")
                speedup = results[key].get('gpu_speedup', 'N/A')
                if isinstance(speedup, float):
                    indicator = "<<<" if speedup > 2 else ""
                    print(f" {speedup:>10.2f}x {indicator}", end="")
                else:
                    print(f" {speedup:>12}", end="")
            print()

    print("-" * 70)
    print("Note: GPU speedup > 2x marked with <<<")
    print("      GPU excels at batch sizes >= 64 or FFT sizes >= 64K")

    return results


def plot_batch_results(results=None, save_path='2_performance_batch.png'):
    """Plot batch benchmark results."""
    if results is None:
        results = benchmark_batch_fft()

    # Extract data for plotting
    sizes = []
    batches = []
    speedups = []

    for key, vals in results.items():
        if 'gpu_speedup' in vals and isinstance(vals['gpu_speedup'], float):
            parts = key.split('x')
            sizes.append(int(parts[0]))
            batches.append(int(parts[1]))
            speedups.append(vals['gpu_speedup'])

    if not speedups:
        print("No GPU data available for plotting")
        return

    # Create heatmap-style plot
    unique_sizes = sorted(set(sizes))
    unique_batches = sorted(set(batches))

    speedup_matrix = np.zeros((len(unique_sizes), len(unique_batches)))
    for s, b, sp in zip(sizes, batches, speedups):
        i = unique_sizes.index(s)
        j = unique_batches.index(b)
        speedup_matrix[i, j] = sp

    plt.figure(figsize=(10, 6))
    plt.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto',
               vmin=0.5, vmax=max(speedups) if speedups else 10)
    plt.colorbar(label='GPU Speedup vs Best CPU')

    plt.xticks(range(len(unique_batches)), [str(b) for b in unique_batches])
    plt.yticks(range(len(unique_sizes)), [f'{s//1024}K' for s in unique_sizes])
    plt.xlabel('Batch Size')
    plt.ylabel('FFT Size')
    plt.title('GPU Speedup vs CPU (green = GPU faster)')

    # Add values to cells
    for i in range(len(unique_sizes)):
        for j in range(len(unique_batches)):
            val = speedup_matrix[i, j]
            color = 'white' if val > 3 else 'black'
            plt.text(j, i, f'{val:.1f}x', ha='center', va='center', color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Batch benchmark plot saved to {save_path}")


# Call the plot function if this script is run directly
if __name__ == "__main__":
    # Print available backends
    available = get_available_backends()
    print(f"Available FFT backends: {available}")

    # Run single FFT benchmarks (existing code)
    plot_fft_results()

    # Run batch benchmarks (new GPU comparison)
    print("\n\nRunning batch FFT benchmark...")
    batch_results = benchmark_batch_fft()
    plot_batch_results(batch_results)

    # Save batch results
    with open('2_performance_batch.json', 'w') as f:
        json.dump(batch_results, f, indent=4)
    print("Batch results saved to 2_performance_batch.json")
