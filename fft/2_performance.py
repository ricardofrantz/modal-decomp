import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from fft_backends import get_fft_func, get_fft_backend_names
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
# Generate powers of two and nearby off values, up to 128k (131072)
powers = list(range(10, 17))  # 2^10 (1024) to 2^17 (131072)
sizes_pow2 = [2 ** p for p in powers]
sizes_off = []
for n in sizes_pow2:
    # Add both +1 and -1, +3 and -3 neighbors, but only if positive, not a power of two, and <= 131072
    for delta in [-3, -1, +1, +3]:
        off_val = n + delta
        if 0 < off_val <= 131072 and (off_val & (off_val - 1)) != 0:
            sizes_off.append(off_val)

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
    backend_names = [k[:-9] for k in data.keys() if k.endswith('_fft_time')]
    plt.figure(figsize=(12, 8))
    fit_results = {}
    for backend in backend_names:
        times = np.array(data[f"{backend}_fft_time"])
        line_plot, = plt.plot(sizes, times, marker='o', label=f"{backend} FFT")
        # Only fit where times > 0 (exclude failed runs)
        valid = times > 0
        if np.sum(valid) > 1:
            log_sizes = np.log(sizes[valid])
            log_times = np.log(times[valid])
            slope, intercept = np.polyfit(log_sizes, log_times, 1)
            fit_results[backend] = (slope, intercept)
            fit_line = np.exp(intercept) * sizes**slope
            fit_plot, = plt.plot(sizes, fit_line, linestyle='--', label=f"{backend} fit (slope={slope:.2f})", color=line_plot.get_color())
            # Annotate slope at the largest size
            x_annot = sizes[-1]
            y_annot = np.exp(intercept) * x_annot**slope
            plt.annotate(f"slope={slope:.2f}",
                         xy=(x_annot, y_annot),
                         xytext=(10, 0),
                         textcoords='offset points',
                         color=line_plot.get_color(),
                         fontsize=10,
                         va='center',
                         fontweight='bold')
    for size in sizes:
        size_int = int(size)
        if size_int & (size_int - 1) != 0:
            plt.axvline(x=size, color='lightgray', linestyle='--', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Signal Size (samples)')
    plt.ylabel('Time (seconds) for 10 iterations')
    plt.title('FFT Performance Comparison Across Libraries (1D Signals)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig('2_performance_performance.png', dpi=DPI, bbox_inches='tight')

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
    for backend in backend_names:
        if backend in fit_results:
            slope, intercept = fit_results[backend]
            print(f"{backend}: slope={slope:.3f}, intercept={intercept:.3f}")
        else:
            print(f"{backend}: insufficient valid data for fit")
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
    print("Plots saved as 2_performance_performance.png, 2_performance_accuracy.png, and 2_performance_signal.png")

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
    print("Plots saved as 2_performance_performance.png, 2_performance_accuracy.png, and 2_performance_signal.png")

# Call the plot function if this script is run directly
if __name__ == "__main__":
    # After running the FFT comparisons and saving results
    plot_fft_results()
