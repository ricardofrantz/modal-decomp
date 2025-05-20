import numpy as np
import cv2
from scipy.fft import fft as scipy_fft
from numpy.fft import fft as numpy_fft
import tensorflow as tf
import torch
import timeit
import json
import matplotlib.pyplot as plt
from scipy import signal

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
    noise_level = 0.15
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
def compare_fft(size):
    # Generate the test signal
    signal = generate_signal(size)
    
    # Dictionary to store results
    result_dict = {}
    
    # Reference implementation (SciPy)
    scipy_fft_time = 0
    scipy_result = None
    
    # Measure SciPy FFT time
    try:
        def scipy_fft_wrapper():
            return scipy_fft(signal)
        
        scipy_fft_time = timeit.timeit(scipy_fft_wrapper, number=10)
        scipy_result = np.abs(scipy_fft_wrapper())
        result_dict['scipy_fft_time'] = scipy_fft_time
    except Exception as e:
        print(f"Error with SciPy FFT: {e}")
        result_dict['scipy_fft_time'] = 0
    
    # If SciPy failed, we can't compare other implementations
    if scipy_result is None:
        return {k: 0 for k in ['scipy_fft_time', 'numpy_fft_time', 'tf_fft_time', 
                              'cv2_fft_time', 'torch_fft_time', 'numpy_error', 
                              'tf_error', 'torch_error', 'cv2_error']}
    
    # NumPy FFT
    try:
        def numpy_fft_wrapper():
            return numpy_fft(signal)
        
        numpy_fft_time = timeit.timeit(numpy_fft_wrapper, number=10)
        numpy_result = np.abs(numpy_fft_wrapper())
        
        result_dict['numpy_fft_time'] = numpy_fft_time
        result_dict['numpy_error'] = float(np.mean(np.abs(scipy_result - numpy_result)))
    except Exception as e:
        print(f"Error with NumPy FFT: {e}")
        result_dict['numpy_fft_time'] = 0
        result_dict['numpy_error'] = 0
    
    # TensorFlow FFT
    try:
        # Convert to TensorFlow tensor
        signal_tf = tf.convert_to_tensor(signal)
        signal_tf_complex = tf.cast(signal_tf, tf.complex64)
        
        def tf_fft_wrapper():
            return tf.signal.fft(signal_tf_complex)
        
        tf_fft_time = timeit.timeit(tf_fft_wrapper, number=10)
        tf_result = np.abs(tf_fft_wrapper().numpy())
        
        result_dict['tf_fft_time'] = tf_fft_time
        result_dict['tf_error'] = float(np.mean(np.abs(scipy_result[:len(tf_result)] - tf_result[:len(scipy_result)])))
    except Exception as e:
        print(f"Error with TensorFlow FFT: {e}")
        result_dict['tf_fft_time'] = 0
        result_dict['tf_error'] = 0
    
    # PyTorch FFT
    try:
        # Convert to PyTorch tensor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        signal_torch = torch.tensor(signal, device=device)
        
        def torch_fft_wrapper():
            return torch.fft.fft(signal_torch)
        
        torch_fft_time = timeit.timeit(torch_fft_wrapper, number=10)
        torch_result = np.abs(torch_fft_wrapper().cpu().numpy())
        
        result_dict['torch_fft_time'] = torch_fft_time
        result_dict['torch_error'] = float(np.mean(np.abs(scipy_result[:len(torch_result)] - torch_result[:len(scipy_result)])))
    except Exception as e:
        print(f"Error with PyTorch FFT: {e}")
        result_dict['torch_fft_time'] = 0
        result_dict['torch_error'] = 0
    
    # OpenCV FFT
    try:
        # OpenCV requires reshaping for 1D FFT
        def cv2_fft_wrapper():
            return cv2.dft(np.float32(signal).reshape(-1, 1), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        cv2_fft_time = timeit.timeit(cv2_fft_wrapper, number=10)
        cv2_output = cv2_fft_wrapper()
        
        # Extract magnitude from complex output
        cv2_result = np.abs(cv2_output[:, 0, 0] + 1j * cv2_output[:, 0, 1])
        
        result_dict['cv2_fft_time'] = cv2_fft_time
        result_dict['cv2_error'] = float(np.mean(np.abs(scipy_result[:len(cv2_result)] - cv2_result[:len(scipy_result)])))
    except Exception as e:
        print(f"Error with OpenCV FFT: {e}")
        result_dict['cv2_fft_time'] = 0
        result_dict['cv2_error'] = 0
    
    return result_dict

# Test with different sizes for 1D signals, including non-powers of 2
# Using a mix of power-of-2 and non-power-of-2 sizes
sizes = [1000, 2049, 3333, 4097, 8199, 10001, 16385, 32798, 65533]

# Initialize results dictionary
results = {
    'scipy_fft_time': [],
    'numpy_fft_time': [],
    'tf_fft_time': [],
    'cv2_fft_time': [],
    'torch_fft_time': [],
    'numpy_error': [],
    'tf_error': [],
    'torch_error': [],
    'cv2_error': []
}

for size in sizes:
    print(f"Processing size: {size}")
    result = compare_fft(size)
    for key in result:
        results[key].append(result[key])
    print(f"Completed size: {size}")

# Save the results to a JSON file
with open('fft_results.json', 'w') as f:
    results_copy = {k: [float(x) for x in v] for k, v in results.items()}
    results_copy['sizes'] = [float(x) for x in sizes]  # Convert sizes to float
    json.dump(results_copy, f, indent=4)
    print("Results saved to fft_results.json")

# Function to plot the results from the JSON file
def plot_fft_results(json_file='fft_results.json'):
    # Load the results from the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    sizes = data['sizes']
    
    # Create a figure for timing comparison
    plt.figure(figsize=(12, 8))
    
    # Plot FFT times for each library
    plt.plot(sizes, data['scipy_fft_time'], 'o-', label='SciPy FFT')
    plt.plot(sizes, data['numpy_fft_time'], 's-', label='NumPy FFT')
    plt.plot(sizes, data['tf_fft_time'], '^-', label='TensorFlow FFT')
    plt.plot(sizes, data['cv2_fft_time'], 'D-', label='OpenCV FFT')
    plt.plot(sizes, data['torch_fft_time'], 'x-', label='PyTorch FFT')
    
    # Mark non-power-of-2 sizes with vertical lines
    for size in sizes:
        # Convert to int for bitwise operation
        size_int = int(size)
        if size_int & (size_int - 1) != 0:  # Check if not a power of 2
            plt.axvline(x=size, color='lightgray', linestyle='--', alpha=0.5)
    
    # Set log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Signal Size (samples)')
    plt.ylabel('Time (seconds) for 10 iterations')
    plt.title('FFT Performance Comparison Across Libraries (1D Signals)')
    
    # Add grid and legend
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save the figure
    plt.savefig('fft_performance_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create a second figure for error comparison
    plt.figure(figsize=(12, 8))
    
    # Plot FFT accuracy (error compared to scipy) for each library
    plt.plot(sizes, data['numpy_error'], 's-', label='NumPy Error')
    plt.plot(sizes, data['tf_error'], '^-', label='TensorFlow Error')
    plt.plot(sizes, data['cv2_error'], 'D-', label='OpenCV Error')
    plt.plot(sizes, data['torch_error'], 'x-', label='PyTorch Error')
    
    # Mark non-power-of-2 sizes with vertical lines
    for size in sizes:
        # Convert to int for bitwise operation
        size_int = int(size)
        if size_int & (size_int - 1) != 0:  # Check if not a power of 2
            plt.axvline(x=size, color='lightgray', linestyle='--', alpha=0.5)
    
    # Set log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Signal Size (samples)')
    plt.ylabel('Mean Absolute Error (vs SciPy)')
    plt.title('FFT Accuracy Comparison Across Libraries (1D Signals)')
    
    # Add grid and legend
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save the figure
    plt.savefig('fft_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create a figure to visualize a sample signal
    plt.figure(figsize=(12, 8))
    
    # Generate a sample signal
    sample_size = 1000
    sample_signal = generate_signal(sample_size)
    t = np.linspace(0, 1, sample_size)
    
    # Plot the signal
    plt.subplot(2, 1, 1)
    plt.plot(t, sample_signal)
    plt.title('Sample 1D Signal with Realistic Noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot the FFT magnitude
    plt.subplot(2, 1, 2)
    
    # Safely compute and plot FFT
    try:
        # Compute FFT
        fft_complex = scipy_fft(sample_signal)
        fft_magnitude = np.abs(fft_complex)
        
        # Calculate frequency bins
        dt = t[1] - t[0]  # Time step
        freqs = np.fft.fftfreq(len(sample_signal), dt)
        
        # Only plot positive frequencies (first half)
        positive_freq_idx = len(freqs) // 2
        
        # Plot magnitude spectrum
        plt.plot(freqs[:positive_freq_idx], fft_magnitude[:positive_freq_idx])
        
        # Add peak markers for the known frequency components
        peak_freqs = [10, 25, 50]  # The frequencies we added to the signal
        for freq in peak_freqs:
            idx = np.argmin(np.abs(freqs[:positive_freq_idx] - freq))
            plt.plot(freqs[idx], fft_magnitude[idx], 'ro')
            plt.text(freqs[idx], fft_magnitude[idx], f"{freq} Hz", 
                     verticalalignment='bottom', horizontalalignment='center')
    except Exception as e:
        print(f"Error plotting FFT: {e}")
        plt.text(0.5, 0.5, f"Error computing FFT: {e}", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    plt.title('FFT Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_signal_and_fft.png', dpi=300, bbox_inches='tight')
    
    print("Plots saved as fft_performance_comparison.png, fft_accuracy_comparison.png, and sample_signal_and_fft.png")

# Call the plot function if this script is run directly
if __name__ == "__main__":
    # After running the FFT comparisons and saving results
    plot_fft_results()
