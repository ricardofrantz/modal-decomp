#!/usr/bin/env python3
"""
2D Benchmark Examples for POD, DMD, and SPOD
============================================

This module provides analytical 2D flow fields commonly used as benchmarks
in modal decomposition research:

1. **Double Gyre** (Shadden et al., 2005)
   - Time-periodic chaotic mixing flow
   - Standard benchmark for Lagrangian Coherent Structures (LCS)
   - Good for DMD (periodic dynamics) and SPOD (spectral content)

2. **Taylor-Green Vortex** (Taylor & Green, 1937)
   - Decaying vortex with exact Navier-Stokes solution
   - Good for POD (energy-based ranking) and DMD (exponential decay)

3. **Cylinder Wake** (von Kármán vortex street)
   - Synthetic periodic shedding at Strouhal number St ≈ 0.16-0.17
   - Standard benchmark for POD/SPOD of periodic flows

References:
-----------
- Shadden et al. (2005): "Definition and properties of Lagrangian coherent
  structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows"
  Physica D, 212(3-4), 271-304.
  https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html

- Taylor & Green (1937): "Mechanism of the production of small eddies from large ones"
  Proc. R. Soc. Lond. A, 158(895), 499-521.
  https://en.wikipedia.org/wiki/Taylor–Green_vortex

- Noack et al. (2003): "A hierarchy of low-dimensional models for the transient
  and post-transient cylinder wake" J. Fluid Mech. 497, 335-363.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# =============================================================================
# Double Gyre Flow (Shadden et al., 2005)
# =============================================================================

def double_gyre(
    Nx: int = 100,
    Ny: int = 50,
    Nt: int = 200,
    A: float = 0.25,
    epsilon: float = 0.25,
    omega: float = 2 * np.pi / 10,  # Period T=10
    x_range: Tuple[float, float] = (0, 2),
    y_range: Tuple[float, float] = (0, 1),
    t_max: float = 20.0,
) -> dict:
    """
    Generate Double Gyre velocity field.

    The Double Gyre is a time-periodic flow with chaotic particle trajectories,
    commonly used as a benchmark for Lagrangian Coherent Structures (LCS) and
    modal decomposition methods.

    Velocity field:
        u = -π A sin(π f(x,t)) cos(π y)
        v =  π A cos(π f(x,t)) sin(π y) df/dx

    where f(x,t) = ε sin(ωt) x² + (1 - 2ε sin(ωt)) x

    Parameters
    ----------
    Nx, Ny : int
        Grid points in x and y directions
    Nt : int
        Number of time snapshots
    A : float
        Amplitude of the gyre (default 0.25)
    epsilon : float
        Oscillation amplitude (default 0.25)
    omega : float
        Angular frequency (default 2π/10 for period T=10)
    x_range, y_range : tuple
        Domain extents (default [0,2] x [0,1])
    t_max : float
        Maximum time (default 20.0)

    Returns
    -------
    dict
        Dictionary with keys: 'u', 'v', 'x', 'y', 't', 'dt', 'Nx', 'Ny', 'Ns',
        'q' (combined field), 'metadata'
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing='ij')  # (Nx, Ny)

    # Preallocate velocity fields
    u = np.zeros((Nt, Nx, Ny))
    v = np.zeros((Nt, Nx, Ny))

    for i, ti in enumerate(t):
        # Time-dependent parameters
        a = epsilon * np.sin(omega * ti)
        b = 1 - 2 * epsilon * np.sin(omega * ti)

        # f(x,t) and its derivative
        f = a * X**2 + b * X
        dfdx = 2 * a * X + b

        # Velocity field
        u[i] = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * Y)
        v[i] = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * Y) * dfdx

    # Flatten spatial dimensions for modal analysis: (Nt, Nx*Ny)
    u_flat = u.reshape(Nt, -1)
    v_flat = v.reshape(Nt, -1)

    # Combined field (u, v stacked)
    q = np.hstack([u_flat, v_flat])  # (Nt, 2*Nx*Ny)

    return {
        'u': u,
        'v': v,
        'x': x,
        'y': y,
        't': t,
        'dt': dt,
        'Nx': Nx,
        'Ny': Ny,
        'Ns': Nt,
        'q': q,
        'q_u': u_flat,  # Just u component
        'q_v': v_flat,  # Just v component
        'metadata': {
            'name': 'Double Gyre',
            'A': A,
            'epsilon': epsilon,
            'omega': omega,
            'period': 2 * np.pi / omega,
        }
    }


# =============================================================================
# Taylor-Green Vortex (Exact Navier-Stokes Solution)
# =============================================================================

def taylor_green_vortex(
    Nx: int = 64,
    Ny: int = 64,
    Nt: int = 100,
    nu: float = 0.01,  # Kinematic viscosity
    U0: float = 1.0,   # Initial velocity amplitude
    L: float = 2 * np.pi,  # Domain size
    t_max: Optional[float] = None,
) -> dict:
    """
    Generate Taylor-Green Vortex velocity field (exact 2D Navier-Stokes solution).

    The Taylor-Green vortex is an unsteady decaying vortex with an exact
    analytical solution to the incompressible Navier-Stokes equations.

    Velocity field:
        u(x,y,t) = -U₀ cos(x) sin(y) exp(-2νt)
        v(x,y,t) =  U₀ sin(x) cos(y) exp(-2νt)
        p(x,y,t) = -ρU₀²/4 (cos(2x) + cos(2y)) exp(-4νt)

    Parameters
    ----------
    Nx, Ny : int
        Grid points in x and y directions
    Nt : int
        Number of time snapshots
    nu : float
        Kinematic viscosity (controls decay rate)
    U0 : float
        Initial velocity amplitude
    L : float
        Domain size (default 2π for one period)
    t_max : float, optional
        Maximum time (default: 3 decay time constants = 3/(2ν))

    Returns
    -------
    dict
        Dictionary with keys: 'u', 'v', 'p', 'x', 'y', 't', 'dt', 'Nx', 'Ny', 'Ns',
        'q' (combined field), 'metadata'
    """
    # Default t_max: 3 decay time constants
    if t_max is None:
        t_max = 3.0 / (2 * nu)

    # Create grid
    x = np.linspace(0, L, Nx, endpoint=False)
    y = np.linspace(0, L, Ny, endpoint=False)
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing='ij')  # (Nx, Ny)

    # Preallocate fields
    u = np.zeros((Nt, Nx, Ny))
    v = np.zeros((Nt, Nx, Ny))
    p = np.zeros((Nt, Nx, Ny))

    # Decay rate
    decay = np.exp(-2 * nu * t)

    for i, ti in enumerate(t):
        d = decay[i]
        u[i] = -U0 * np.cos(X) * np.sin(Y) * d
        v[i] = U0 * np.sin(X) * np.cos(Y) * d
        p[i] = -0.25 * U0**2 * (np.cos(2*X) + np.cos(2*Y)) * d**2

    # Flatten spatial dimensions
    u_flat = u.reshape(Nt, -1)
    v_flat = v.reshape(Nt, -1)
    p_flat = p.reshape(Nt, -1)

    # Combined field
    q = np.hstack([u_flat, v_flat])

    # Theoretical decay eigenvalue for DMD
    lambda_theory = np.exp(-2 * nu * dt)

    return {
        'u': u,
        'v': v,
        'p': p,
        'x': x,
        'y': y,
        't': t,
        'dt': dt,
        'Nx': Nx,
        'Ny': Ny,
        'Ns': Nt,
        'q': q,
        'q_u': u_flat,
        'q_v': v_flat,
        'q_p': p_flat,
        'metadata': {
            'name': 'Taylor-Green Vortex',
            'nu': nu,
            'U0': U0,
            'L': L,
            'decay_rate': 2 * nu,
            'dmd_eigenvalue': lambda_theory,
        }
    }


# =============================================================================
# Synthetic Cylinder Wake (von Kármán Street)
# =============================================================================

def cylinder_wake(
    Nx: int = 100,
    Ny: int = 50,
    Nt: int = 500,
    Re: float = 100,
    D: float = 1.0,  # Cylinder diameter
    U_inf: float = 1.0,  # Freestream velocity
    x_range: Tuple[float, float] = (0, 10),
    y_range: Tuple[float, float] = (-2.5, 2.5),
) -> dict:
    """
    Generate synthetic cylinder wake (von Kármán vortex street).

    This creates a simplified model of the cylinder wake with:
    - Periodic vortex shedding at Strouhal number St ≈ 0.16-0.17
    - Exponential wake decay downstream
    - Alternating vortices above/below centerline

    Parameters
    ----------
    Nx, Ny : int
        Grid points in x and y directions
    Nt : int
        Number of time snapshots
    Re : float
        Reynolds number (default 100, vortex shedding regime)
    D : float
        Cylinder diameter
    U_inf : float
        Freestream velocity
    x_range, y_range : tuple
        Domain extents

    Returns
    -------
    dict
        Dictionary with velocity fields and metadata
    """
    # Strouhal number (empirical fit for Re ~ 40-200)
    # Roshko (1954): St ≈ 0.212(1 - 21.2/Re) for 40 < Re < 150
    # Simplified: St ≈ 0.167 at Re=100
    St = 0.212 * (1 - 21.2 / Re)  # Gives St ≈ 0.167 at Re=100

    # Shedding frequency
    f_shed = St * U_inf / D
    omega = 2 * np.pi * f_shed

    # Time array (capture ~10 shedding cycles)
    T_shed = 1.0 / f_shed
    t_max = 10 * T_shed
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]

    # Create grid
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Wake parameters
    x_cyl = 1.0  # Cylinder center
    wake_width = D * (1 + 0.1 * np.sqrt(np.maximum(X - x_cyl, 0)))  # Spreading wake
    wake_decay = np.exp(-0.1 * np.maximum(X - x_cyl, 0))  # Downstream decay

    # Preallocate
    u = np.zeros((Nt, Nx, Ny))
    v = np.zeros((Nt, Nx, Ny))

    # Amplitude of fluctuations (stronger near cylinder)
    amp = 0.3 * U_inf * wake_decay

    for i, ti in enumerate(t):
        # Phase of vortex shedding
        phase = omega * ti

        # Transverse (y) oscillation - von Kármán street pattern
        # Spatial modulation: vortices convect downstream
        k_x = omega / (0.8 * U_inf)  # Convection wavenumber
        spatial_phase = k_x * (X - x_cyl)

        # Gaussian envelope in y
        y_envelope = np.exp(-(Y**2) / (2 * wake_width**2))

        # Velocity fluctuations
        u[i] = U_inf * (1 - 0.5 * wake_decay * y_envelope)  # Base flow with deficit
        u[i] += amp * np.sin(phase - spatial_phase) * y_envelope * (Y / wake_width)

        v[i] = amp * np.cos(phase - spatial_phase) * y_envelope

    # Add small noise for realism (seeded for reproducibility)
    rng = np.random.default_rng(seed=42)
    noise_level = 0.02 * U_inf
    u += rng.standard_normal(u.shape) * noise_level
    v += rng.standard_normal(v.shape) * noise_level

    # Flatten
    u_flat = u.reshape(Nt, -1)
    v_flat = v.reshape(Nt, -1)
    q = np.hstack([u_flat, v_flat])

    return {
        'u': u,
        'v': v,
        'x': x,
        'y': y,
        't': t,
        'dt': dt,
        'Nx': Nx,
        'Ny': Ny,
        'Ns': Nt,
        'q': q,
        'q_u': u_flat,
        'q_v': v_flat,
        'metadata': {
            'name': 'Cylinder Wake',
            'Re': Re,
            'D': D,
            'U_inf': U_inf,
            'St': St,
            'f_shed': f_shed,
            'T_shed': T_shed,
        }
    }


# =============================================================================
# Helper: Create data loader for pyModal
# =============================================================================

def make_loader(data: dict):
    """
    Create a data loader function compatible with pyModal analyzers.

    Parameters
    ----------
    data : dict
        Output from double_gyre(), taylor_green_vortex(), or cylinder_wake()

    Returns
    -------
    callable
        Loader function that returns data in pyModal format
    """
    def loader(file_path):
        return {
            'q': data['q'],
            'x': data['x'],
            'y': data['y'],
            'z': None,
            'dt': data['dt'],
            'Nx': data['Nx'],
            'Ny': data['Ny'],
            'Nz': 1,
            'Ns': data['Ns'],
            'metadata': data['metadata'],
        }
    return loader


# =============================================================================
# 2D Mode Visualization
# =============================================================================

def plot_2d_modes(data: dict, modes: np.ndarray, prefix: str, n_modes: int = 4,
                  title: str = "Modes", component: str = 'both', noise_threshold: float = 1e-6):
    """
    Plot modes reshaped to 2D spatial field.

    Parameters
    ----------
    data : dict
        Original data dict with Nx, Ny, x, y
    modes : np.ndarray
        Mode array of shape (Nspace, n_modes) or (2*Nspace, n_modes)
    prefix : str
        Filename prefix
    n_modes : int
        Number of modes to plot
    component : str
        'u', 'v', or 'both'
    noise_threshold : float
        Skip plotting modes with max amplitude below this threshold (relative to mode 1)
    """
    Nx, Ny = data['Nx'], data['Ny']
    x, y = data['x'], data['y']
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Guard for empty modes array
    if modes.size == 0 or modes.shape[1] == 0:
        print(f"  Warning: No modes to plot for {prefix}")
        return None

    n_modes = min(n_modes, modes.shape[1])

    # Check if modes contain both u and v (stacked)
    if modes.shape[0] == 2 * Nx * Ny:
        # Split into u and v components
        modes_u = modes[:Nx*Ny, :]
        modes_v = modes[Nx*Ny:, :]
        has_both = True
    else:
        modes_u = modes
        modes_v = None
        has_both = False

    # Determine which modes are significant (not numerical noise)
    # Compare to mode 1 amplitude (guard against zero amplitude)
    mode1_amp = np.abs(modes_u[:, 0]).max()
    if mode1_amp == 0 or not np.isfinite(mode1_amp):
        mode1_amp = 1.0  # Fallback to avoid division issues
    significant_modes = []
    for i in range(n_modes):
        mode_amp = np.abs(modes_u[:, i]).max()
        if mode_amp > noise_threshold * mode1_amp:
            significant_modes.append(i)

    if len(significant_modes) == 0:
        print(f"  Warning: No significant modes to plot for {prefix}")
        return None

    n_modes = len(significant_modes)

    # Create figure
    if has_both and component == 'both':
        fig, axes = plt.subplots(2, n_modes, figsize=(4*n_modes, 8))
        if n_modes == 1:
            axes = axes.reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, n_modes, figsize=(4*n_modes, 4))
        if n_modes == 1:
            axes = [axes]

    for plot_idx, mode_idx in enumerate(significant_modes):
        # Get mode and reshape to 2D
        mode_u = np.real(modes_u[:, mode_idx].reshape(Nx, Ny))

        if has_both and component == 'both':
            mode_v = np.real(modes_v[:, mode_idx].reshape(Nx, Ny))

            # U component
            ax = axes[0, plot_idx] if n_modes > 1 else axes[0, 0]
            vmax = np.abs(mode_u).max()
            im = ax.pcolormesh(X, Y, mode_u, shading='auto', cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
            ax.set_title(f"Mode {mode_idx+1} (u)")
            ax.set_aspect('equal')
            if plot_idx == 0:
                ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, shrink=0.8)

            # V component
            ax = axes[1, plot_idx] if n_modes > 1 else axes[1, 0]
            vmax = np.abs(mode_v).max()
            im = ax.pcolormesh(X, Y, mode_v, shading='auto', cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
            ax.set_title(f"Mode {mode_idx+1} (v)")
            ax.set_xlabel('x')
            ax.set_aspect('equal')
            if plot_idx == 0:
                ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax = axes[plot_idx] if n_modes > 1 else axes[0]
            vmax = np.abs(mode_u).max()
            im = ax.pcolormesh(X, Y, mode_u, shading='auto', cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
            ax.set_title(f"Mode {mode_idx+1}")
            ax.set_xlabel('x')
            ax.set_aspect('equal')
            if plot_idx == 0:
                ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path = f"./figs_examples/{prefix}_modes_2d.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)

    return fig


# =============================================================================
# Demonstration: Run POD, DMD, SPOD on all benchmarks
# =============================================================================

def demo_pod(data: dict, n_modes: int = 10, plot: bool = True, prefix: str = None):
    """Run POD analysis on 2D data."""
    from pymodal import PODAnalyzer

    # Use case name as prefix for unique filenames
    if prefix is None:
        prefix = data['metadata']['name'].lower().replace(' ', '_').replace('-', '_')

    loader = make_loader(data)
    analyzer = PODAnalyzer(
        file_path=prefix,
        results_dir="./results_examples",
        figures_dir="./figs_examples",
        n_modes_save=n_modes,
        data_loader=loader,
    )
    # Use run_analysis for full POD with plotting
    analyzer.run_analysis(
        plot_n_modes_spatial=min(4, n_modes),
        plot_n_coeffs_time=min(5, n_modes),
        check_orthogonality=False,
    )

    print(f"\n  POD Results for {data['metadata']['name']}:")
    print(f"    Modes computed: {len(analyzer.eigenvalues)}")
    if len(analyzer.eigenvalues) > 0:
        # Compute cumulative energy
        total_energy = np.sum(analyzer.eigenvalues)
        cum_energy = np.cumsum(analyzer.eigenvalues) / total_energy
        print(f"    Energy in {min(n_modes, len(analyzer.eigenvalues))} modes: {cum_energy[min(n_modes-1, len(cum_energy)-1)]:.2%}")
        print(f"    Top 3 eigenvalues: {analyzer.eigenvalues[:3]}")

        # Plot 2D modes (in addition to default plots)
        if plot and analyzer.modes is not None:
            plot_2d_modes(data, analyzer.modes, f"{prefix}_pod", n_modes=4,
                          title=f"POD Modes - {data['metadata']['name']}", component='both')

    return analyzer


def demo_dmd(data: dict, n_modes: int = 10, plot: bool = True, prefix: str = None):
    """Run DMD analysis on 2D data."""
    from pymodal import DMDAnalyzer

    # Use case name as prefix for unique filenames
    if prefix is None:
        prefix = data['metadata']['name'].lower().replace(' ', '_').replace('-', '_')

    loader = make_loader(data)
    analyzer = DMDAnalyzer(
        file_path=prefix,
        results_dir="./results_examples",
        figures_dir="./figs_examples",
        n_modes_save=n_modes,
        data_loader=loader,
    )
    # Load data first
    analyzer.load_and_preprocess()
    # Perform DMD analysis
    analyzer.perform_dmd()
    # Save results
    analyzer.save_results()

    # Extract frequencies
    dt = data['dt']

    print(f"\n  DMD Results for {data['metadata']['name']}:")
    print(f"    Modes computed: {len(analyzer.eigenvalues)}")

    if len(analyzer.eigenvalues) > 0:
        freqs = np.angle(analyzer.eigenvalues) / (2 * np.pi * dt)
        print(f"    Top eigenvalue magnitudes: {np.abs(analyzer.eigenvalues[:3])}")
        print(f"    Top frequencies (Hz): {freqs[:3]}")

        # Check against known values if available
        meta = data['metadata']
        if 'f_shed' in meta:
            print(f"    Expected shedding freq: {meta['f_shed']:.4f} Hz")
        if 'dmd_eigenvalue' in meta:
            print(f"    Expected eigenvalue: {meta['dmd_eigenvalue']:.6f}")
        if 'omega' in meta:
            expected_freq = meta['omega'] / (2 * np.pi)
            print(f"    Expected gyre freq: {expected_freq:.4f} Hz")

        # Generate all DMD plots
        if plot:
            analyzer.plot_eigenvalues()
            analyzer.plot_modes(plot_n_modes=min(4, n_modes))
            analyzer.plot_time_coefficients(n_coeffs_to_plot=min(4, n_modes))
            analyzer.plot_cumulative_energy()
            analyzer.plot_reconstruction_error()
            # Plot 2D modes (both u and v components)
            plot_2d_modes(data, analyzer.modes, f"{prefix}_dmd", n_modes=4,
                          title=f"DMD Modes - {data['metadata']['name']}",
                          component='both')

    return analyzer


def demo_spod(data: dict, nfft: int = 128, plot: bool = True, prefix: str = None):
    """Run SPOD analysis on 2D data."""
    from pymodal import SPODAnalyzer

    # Use case name as prefix for unique filenames
    if prefix is None:
        prefix = data['metadata']['name'].lower().replace(' ', '_').replace('-', '_')

    # Use just u-component for SPOD (single variable)
    data_spod = data.copy()
    data_spod['q'] = data['q_u']

    loader = make_loader(data_spod)
    analyzer = SPODAnalyzer(
        file_path=prefix,
        nfft=nfft,
        overlap=0.5,
        results_dir="./results_examples",
        figures_dir="./figs_examples",
        data_loader=loader,
    )
    analyzer.run()
    analyzer.perform_spod()  # Compute eigenvalues and modes

    print(f"\n  SPOD Results for {data['metadata']['name']}:")
    print(f"    Frequency bins: {len(analyzer.freq)}")
    print(f"    Blocks used: {analyzer.nblocks}")

    # Find peak frequency
    if len(analyzer.eigenvalues) > 0:
        mode0_energy = np.array([analyzer.eigenvalues[f][0] for f in range(len(analyzer.freq))])
        peak_idx = np.argmax(mode0_energy[1:]) + 1  # Skip DC
        peak_freq = analyzer.freq[peak_idx]
        print(f"    Peak frequency: {peak_freq:.4f} Hz")

        # Check against known values
        meta = data['metadata']
        if 'f_shed' in meta:
            print(f"    Expected shedding freq: {meta['f_shed']:.4f} Hz")
        if 'omega' in meta:
            expected_freq = meta['omega'] / (2 * np.pi)
            print(f"    Expected gyre freq: {expected_freq:.4f} Hz")

        # Generate SPOD plots
        if plot:
            try:
                # Eigenvalue spectrum (energy vs frequency)
                analyzer.plot_eigenvalues_v2()
                # Plot modes at peak frequency
                analyzer.plot_modes(freqs_to_plot=[peak_freq], plot_n_modes=4)
                # Cumulative energy
                analyzer.plot_cumulative_energy(freq_idx=peak_idx)
            except Exception as e:
                print(f"    Warning: Some SPOD plots failed: {e}")

            # Plot 2D SPOD modes at peak frequency
            if hasattr(analyzer, 'modes') and analyzer.modes is not None:
                try:
                    # SPOD modes shape: (n_freq, Nspace, n_blocks)
                    peak_modes = analyzer.modes[peak_idx]  # (Nspace, n_blocks)
                    n_modes_plot = min(4, peak_modes.shape[1])
                    plot_2d_modes(data_spod, peak_modes[:, :n_modes_plot],
                                  f"{prefix}_spod_f{peak_freq:.3f}",
                                  n_modes=n_modes_plot,
                                  title=f"SPOD Modes @ f={peak_freq:.3f} Hz - {data['metadata']['name']}")
                except Exception as e:
                    print(f"    Warning: SPOD 2D mode plot failed: {e}")

    return analyzer


def plot_snapshot(data: dict, t_idx: int = 0, save_path: Optional[str] = None):
    """Plot a snapshot of the 2D velocity field."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    u = data['u'][t_idx]
    v = data['v'][t_idx]
    x, y = data['x'], data['y']
    X, Y = np.meshgrid(x, y, indexing='ij')

    # U velocity
    im0 = axes[0].pcolormesh(X, Y, u, shading='auto', cmap='RdBu_r')
    axes[0].set_title(f"u velocity (t={data['t'][t_idx]:.2f})")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])

    # V velocity
    im1 = axes[1].pcolormesh(X, Y, v, shading='auto', cmap='RdBu_r')
    axes[1].set_title(f"v velocity (t={data['t'][t_idx]:.2f})")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(data['metadata']['name'], fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_velocity_magnitude(data: dict, t_indices: list = None, save_path: Optional[str] = None):
    """Plot velocity magnitude at multiple time steps."""
    if t_indices is None:
        # 4 evenly spaced snapshots
        t_indices = [0, len(data['t'])//4, len(data['t'])//2, 3*len(data['t'])//4]

    n = len(t_indices)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    x, y = data['x'], data['y']
    X, Y = np.meshgrid(x, y, indexing='ij')

    vmax = 0
    for t_idx in t_indices:
        mag = np.sqrt(data['u'][t_idx]**2 + data['v'][t_idx]**2)
        vmax = max(vmax, mag.max())

    for ax, t_idx in zip(axes, t_indices):
        u = data['u'][t_idx]
        v = data['v'][t_idx]
        mag = np.sqrt(u**2 + v**2)

        im = ax.pcolormesh(X, Y, mag, shading='auto', cmap='viridis', vmin=0, vmax=vmax)
        ax.set_title(f"t = {data['t'][t_idx]:.2f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # Add streamlines
        skip = max(1, len(x)//15)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  u[::skip, ::skip], v[::skip, ::skip],
                  color='white', alpha=0.6, scale=vmax*10)

    fig.suptitle(f"{data['metadata']['name']} - Velocity Magnitude", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_time_series(data: dict, probe_points: list = None, save_path: Optional[str] = None):
    """Plot time series at probe points."""
    if probe_points is None:
        # Default: center and quarter points
        nx, ny = data['Nx'], data['Ny']
        probe_points = [(nx//2, ny//2), (nx//4, ny//2), (3*nx//4, ny//2)]

    fig, axes = plt.subplots(len(probe_points), 1, figsize=(10, 3*len(probe_points)), sharex=True)
    if len(probe_points) == 1:
        axes = [axes]

    t = data['t']

    for ax, (ix, iy) in zip(axes, probe_points):
        u_probe = data['u'][:, ix, iy]
        v_probe = data['v'][:, ix, iy]

        ax.plot(t, u_probe, 'b-', label='u', alpha=0.8)
        ax.plot(t, v_probe, 'r-', label='v', alpha=0.8)
        ax.set_ylabel(f'Velocity\n(x={data["x"][ix]:.2f}, y={data["y"][iy]:.2f})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    fig.suptitle(f"{data['metadata']['name']} - Time Series", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# =============================================================================
# Main: Run all demonstrations
# =============================================================================

if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving

    os.makedirs("./results_examples", exist_ok=True)
    os.makedirs("./figs_examples", exist_ok=True)

    print("=" * 70)
    print(" 2D Benchmark Examples for POD, DMD, SPOD")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Double Gyre
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 1. DOUBLE GYRE (Shadden et al., 2005)")
    print("=" * 70)
    print("    Time-periodic chaotic mixing flow")
    print("    Domain: [0,2] x [0,1], Period T=10")

    gyre = double_gyre(Nx=80, Ny=40, Nt=200)

    # Generate figures
    plot_velocity_magnitude(gyre, save_path="./figs_examples/double_gyre_evolution.png")
    plot_time_series(gyre, save_path="./figs_examples/double_gyre_timeseries.png")
    plt.close('all')

    # Run analysis
    demo_pod(gyre, n_modes=10)
    plt.close('all')
    demo_dmd(gyre, n_modes=10)
    demo_spod(gyre, nfft=64)
    plt.close('all')

    # -------------------------------------------------------------------------
    # 2. Taylor-Green Vortex
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 2. TAYLOR-GREEN VORTEX (Exact NS Solution)")
    print("=" * 70)
    print("    Decaying vortex, analytical solution")
    print("    Decay rate: 2*nu, Domain: [0, 2pi]^2")

    tgv = taylor_green_vortex(Nx=64, Ny=64, Nt=100, nu=0.01)

    # Generate figures
    plot_velocity_magnitude(tgv, save_path="./figs_examples/taylor_green_evolution.png")
    plot_time_series(tgv, save_path="./figs_examples/taylor_green_timeseries.png")
    plt.close('all')

    # Run analysis (all three methods)
    demo_pod(tgv, n_modes=5)
    plt.close('all')
    demo_dmd(tgv, n_modes=5)
    plt.close('all')
    demo_spod(tgv, nfft=32)  # Smaller nfft due to only 100 snapshots
    plt.close('all')

    # -------------------------------------------------------------------------
    # 3. Cylinder Wake
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 3. CYLINDER WAKE (von Karman Street)")
    print("=" * 70)
    print("    Synthetic vortex shedding, Re=100")
    print("    Expected Strouhal number: St ~ 0.167")

    wake = cylinder_wake(Nx=100, Ny=50, Nt=500, Re=100)

    # Generate figures
    plot_velocity_magnitude(wake, save_path="./figs_examples/cylinder_wake_evolution.png")
    plot_time_series(wake, save_path="./figs_examples/cylinder_wake_timeseries.png")
    plt.close('all')

    # Run analysis
    demo_pod(wake, n_modes=10)
    plt.close('all')
    demo_dmd(wake, n_modes=10)
    demo_spod(wake, nfft=128)
    plt.close('all')

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print("""
    Figures saved to ./figs_examples/ (24 total):

    For each case (double_gyre, taylor_green_vortex, cylinder_wake):
      - {case}_evolution.png           - Velocity magnitude snapshots
      - {case}_timeseries.png          - Probe time series
      - {case}_pod_eigenvalues.png     - POD eigenvalue spectrum
      - {case}_pod_modes_grid.png      - Spatial POD modes
      - {case}_pod_time_coeffs.png     - Temporal coefficients + PSD
      - {case}_pod_cumulative_energy.png - Cumulative energy
      - {case}_pod_reconstruction_error.png - Reconstruction error
      - {case}_dmd_eigenvalues.png     - DMD eigenvalues (complex plane)

    Results saved to ./results_examples/ (HDF5 files)

    Benchmarks:
    -----------------------------------------------------------
    | Case           | Key Feature        | Validation        |
    |----------------|--------------------|--------------------|
    | Double Gyre    | Period T=10        | DMD freq = 0.1 Hz |
    | Taylor-Green   | Decay rate 2*nu    | DMD |lambda| < 1   |
    | Cylinder Wake  | St ~ 0.167         | SPOD peak @ St    |
    -----------------------------------------------------------
    """)

    print("\n Done! Check ./figs_examples/ for figures.")
