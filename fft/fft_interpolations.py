import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator, interp1d, splrep, splev
from scipy.fft import rfft, rfftfreq
from complex_signal import generate_complex_signal

def variable_time_steps(T, base_dt, variability=0.1):
    times = [0]
    while times[-1] < T:
        # Generate a new time step, ensure it's positive and adds a new point within T
        next_time = times[-1] + abs(np.random.normal(base_dt, variability * base_dt))
        if next_time < T:
            times.append(next_time)
        else:
            # If the next_time would exceed T, break the loop
            break
    return np.array(times)

def periodogram_rfft(x, dt):
    N = len(x)
    X = rfft(x)
    freqs = rfftfreq(N, dt)
    psd = np.abs(X) ** 2 * dt / N
    return freqs, psd

def compare_interpolations_and_ffts(time_original, data_original, time_new):
    freq_orig, fft_orig = periodogram_rfft(data_original, time_original[1] - time_original[0])

    methods = {
        'Linear': interp1d(time_original, data_original, kind='linear'), 
        'Slinear': interp1d(time_original, data_original, kind='slinear'), # first order spiline
        'Zero': interp1d(time_original, data_original, kind='zero'), # zero order spline
        'Nearest': interp1d(time_original, data_original, kind='nearest'), # snap to nearest value
        'Cubic Spline': CubicSpline(time_original, data_original),
        'Quintic Spline': lambda x: splev(x, splrep(time_original, data_original, k=5)),
        'Akima': Akima1DInterpolator(time_original, data_original, method='akima'),
        'Makima': Akima1DInterpolator(time_original, data_original, method='makima'),

    }

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Comparison of Interpolation Methods and Their FFTs', fontsize=16)

    mse_scores = {}

    # Plot interpolated signals
    axs[0].plot(time_original, data_original, 'ko-', label='Original', markersize=3)
    for name, interpolator in methods.items():
        data_interp = interpolator(time_new) if callable(interpolator) else interpolator(time_new)
        axs[0].plot(time_new, data_interp, label=name)

        # Compute FFT and MSE
        freq_new, fft_new = periodogram_rfft(data_interp, time_new[1] - time_new[0])
        mse = np.mean((np.interp(freq_new, freq_orig, fft_orig) - fft_new) ** 2)
        mse_scores[name] = mse  # Store MSE score

        axs[1].semilogy(freq_new, fft_new, label=f'{name} (MSE: {mse:.2e})')

    axs[0].set_title('Interpolated Signals')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].set_xlim(1, 2)


    axs[1].semilogy(freq_orig, fft_orig, 'k--', label='Original')
    axs[1].set_title('FFT of Interpolated Signals')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Power Spectral Density')
    axs[1].legend()
    axs[1].set_xlim(1e-2, 1)
    axs[1].set_ylim(1e-6, 1e2)

    plt.tight_layout()
    plt.savefig('interpolation_fft_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Sort methods by MSE scores in ascending order
    sorted_methods = sorted(mse_scores.items(), key=lambda x: x[1])
    
    print("Interpolation methods ranked from best to worst based on MSE of their FFTs:")
    for rank, (method, mse) in enumerate(sorted_methods, 1):
        print(f"{rank}. {method} - MSE: {mse:.4e}")

    return sorted_methods

# Parameters based on your original settings
L = 1.0  # Characteristic length
U = 1.0  # Characteristic velocity
St1 = 0.1212131  # Lowest Strouhal number
St2 = 0.0874888  # Calculated irrational Strouhal number
num_harmonics_f1 = 5
num_harmonics_f2 = 3

periods = 10
T = periods / St2
dt_orig = 0.00043231321123124  # Original time step
t_orig = np.arange(0, T, dt_orig)
# t_orig = variable_time_steps(T, dt_orig, variability=0.12)
x_orig = generate_complex_signal(t_orig, St1, St2, num_harmonics_f1=num_harmonics_f1, num_harmonics_f2=num_harmonics_f2, noise_level=0.1)

dt_new = 0.05  # New time step
t_new = np.arange(0, T, dt_new)
mse_results = compare_interpolations_and_ffts(t_orig, x_orig, t_new)
