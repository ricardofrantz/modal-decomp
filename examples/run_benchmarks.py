#!/usr/bin/env python3
"""
Unified Benchmark Suite for pyModal.

Runs modal decomposition methods (POD, DMD, SPOD, BSMD, ST-POD) on:
- Analytical test cases (generated on-the-fly)
- Experimental/simulation datasets (from data/)

Results and figures are organized by case:
    results/{case_name}/{method}/
    figures/{case_name}/{method}/

Usage:
    # Run all benchmarks (analytical + experimental)
    uv run python examples/run_benchmarks.py

    # Analytical benchmarks only (double gyre, Taylor-Green, cylinder wake)
    uv run python examples/run_benchmarks.py --analytical

    # Experimental data only
    uv run python examples/run_benchmarks.py --experimental

    # Specific cases
    uv run python examples/run_benchmarks.py --cases cavity,jet

    # Specific methods
    uv run python examples/run_benchmarks.py --methods pod,dmd,spod

    # Quick mode (smaller grids for analytical cases)
    uv run python examples/run_benchmarks.py --quick --analytical

Author: R. Frantz
"""

import argparse
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pymodal import BSMDAnalyzer, DMDAnalyzer, PODAnalyzer, SPODAnalyzer, STPODAnalyzer

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


# =============================================================================
# Analytical Test Cases (generated on-the-fly)
# =============================================================================


def generate_double_gyre(
    Nx: int = 80, Ny: int = 40, Nt: int = 200, quick: bool = False
) -> dict:
    """Generate Double Gyre velocity field (Shadden et al., 2005)."""
    if quick:
        Nx, Ny, Nt = 40, 20, 100

    A, epsilon, omega = 0.25, 0.25, 2 * np.pi / 10
    t_max = 20.0

    x = np.linspace(0, 2, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.zeros((Nt, Nx, Ny))
    v = np.zeros((Nt, Nx, Ny))

    for i, ti in enumerate(t):
        a = epsilon * np.sin(omega * ti)
        b = 1 - 2 * epsilon * np.sin(omega * ti)
        f = a * X**2 + b * X
        dfdx = 2 * a * X + b
        u[i] = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * Y)
        v[i] = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * Y) * dfdx

    # Use u component only (consistent with spatial weight dimensions)
    q = u.reshape(Nt, -1)

    return {
        "q": q,
        "x": x,
        "y": y,
        "z": None,
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": 1,
        "Ns": Nt,
        "metadata": {
            "name": "Double Gyre",
            "period": 2 * np.pi / omega,
            "expected_freq": omega / (2 * np.pi),
        },
    }


def generate_taylor_green(
    Nx: int = 64, Ny: int = 64, Nt: int = 100, quick: bool = False
) -> dict:
    """Generate Taylor-Green Vortex (exact NS solution)."""
    if quick:
        Nx, Ny, Nt = 32, 32, 50

    nu, U0, L = 0.01, 1.0, 2 * np.pi
    t_max = 3.0 / (2 * nu)

    x = np.linspace(0, L, Nx, endpoint=False)
    y = np.linspace(0, L, Ny, endpoint=False)
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.zeros((Nt, Nx, Ny))
    v = np.zeros((Nt, Nx, Ny))
    decay = np.exp(-2 * nu * t)

    for i, ti in enumerate(t):
        d = decay[i]
        u[i] = -U0 * np.cos(X) * np.sin(Y) * d
        v[i] = U0 * np.sin(X) * np.cos(Y) * d

    # Use u component only (consistent with spatial weight dimensions)
    q = u.reshape(Nt, -1)

    return {
        "q": q,
        "x": x,
        "y": y,
        "z": None,
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": 1,
        "Ns": Nt,
        "metadata": {
            "name": "Taylor-Green Vortex",
            "decay_rate": 2 * nu,
            "dmd_eigenvalue": np.exp(-2 * nu * dt),
        },
    }


def generate_cylinder_wake(
    Nx: int = 100, Ny: int = 50, Nt: int = 500, quick: bool = False
) -> dict:
    """Generate synthetic cylinder wake (von Kármán street)."""
    if quick:
        Nx, Ny, Nt = 50, 25, 200

    Re, D, U_inf = 100, 1.0, 1.0
    St = 0.212 * (1 - 21.2 / Re)
    f_shed = St * U_inf / D
    omega = 2 * np.pi * f_shed
    T_shed = 1.0 / f_shed
    t_max = 10 * T_shed

    x = np.linspace(0, 10, Nx)
    y = np.linspace(-2.5, 2.5, Ny)
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    x_cyl = 1.0
    wake_width = D * (1 + 0.1 * np.sqrt(np.maximum(X - x_cyl, 0)))
    wake_decay = np.exp(-0.1 * np.maximum(X - x_cyl, 0))
    amp = 0.3 * U_inf * wake_decay

    u = np.zeros((Nt, Nx, Ny))
    v = np.zeros((Nt, Nx, Ny))

    for i, ti in enumerate(t):
        phase = omega * ti
        k_x = omega / (0.8 * U_inf)
        spatial_phase = k_x * (X - x_cyl)
        y_envelope = np.exp(-(Y**2) / (2 * wake_width**2))
        u[i] = U_inf * (1 - 0.5 * wake_decay * y_envelope)
        u[i] += amp * np.sin(phase - spatial_phase) * y_envelope * (Y / wake_width)
        v[i] = amp * np.cos(phase - spatial_phase) * y_envelope

    # Add noise
    rng = np.random.default_rng(seed=42)
    u += rng.standard_normal(u.shape) * 0.02 * U_inf
    v += rng.standard_normal(v.shape) * 0.02 * U_inf

    # Use u component only (consistent with spatial weight dimensions)
    q = u.reshape(Nt, -1)

    return {
        "q": q,
        "x": x,
        "y": y,
        "z": None,
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": 1,
        "Ns": Nt,
        "metadata": {"name": "Cylinder Wake", "Re": Re, "St": St, "f_shed": f_shed},
    }


# =============================================================================
# Case Registry
# =============================================================================

# Analytical cases (generated on-the-fly)
ANALYTICAL_CASES = {
    "double_gyre": {
        "generator": generate_double_gyre,
        "weight_type": "uniform",
        "nfft": 64,
        "embedding_dim": 20,
        "description": "Double Gyre (time-periodic mixing flow)",
    },
    "taylor_green": {
        "generator": generate_taylor_green,
        "weight_type": "uniform",
        "nfft": 32,
        "embedding_dim": 10,
        "description": "Taylor-Green Vortex (decaying exact NS solution)",
    },
    "cylinder_wake": {
        "generator": generate_cylinder_wake,
        "weight_type": "uniform",
        "nfft": 128,
        "embedding_dim": 20,
        "description": "Cylinder Wake (synthetic von Kármán street)",
    },
}

# Experimental/simulation cases (from data files)
EXPERIMENTAL_CASES = {
    "cavity": {
        "file": "cavity/cavityPIV.mat",
        "weight_type": "uniform",
        "nfft": 256,
        "embedding_dim": 20,
        "description": "Cavity PIV (4000 snapshots, 260 spatial)",
    },
    "jet_small": {
        "file": "jet/jetLES_small.mat",
        "weight_type": "polar",
        "nfft": 128,
        "embedding_dim": 10,
        "description": "Jet LES small (1000 snapshots, 1760 spatial)",
    },
    "jet": {
        "file": "jet/jetLES.mat",
        "weight_type": "polar",
        "nfft": 256,
        "embedding_dim": 20,
        "description": "Jet LES full (5000 snapshots, 6825 spatial)",
    },
    "dnamix_1": {
        "file": "dnamix/snp1-947_u.npz",
        "weight_type": "uniform",
        "nfft": 128,
        "embedding_dim": 15,
        "description": "DNamiX snapshots 1-947 (947 snapshots, 40800 spatial)",
    },
    "dnamix_2": {
        "file": "dnamix/snp948-1894_u.npz",
        "weight_type": "uniform",
        "nfft": 128,
        "embedding_dim": 15,
        "description": "DNamiX snapshots 948-1894",
    },
    "dnamix_3": {
        "file": "dnamix/snp1895-2841_u.npz",
        "weight_type": "uniform",
        "nfft": 128,
        "embedding_dim": 15,
        "description": "DNamiX snapshots 1895-2841",
    },
    "dnamix_4": {
        "file": "dnamix/snp2842-3000_u.npz",
        "weight_type": "uniform",
        "nfft": 64,
        "embedding_dim": 10,
        "description": "DNamiX snapshots 2842-3000 (159 snapshots)",
    },
}

# Quick mode subsets
QUICK_ANALYTICAL = ["double_gyre", "taylor_green"]
QUICK_EXPERIMENTAL = ["cavity", "jet_small"]

ALL_METHODS = ["pod", "dmd", "spod", "bsmd", "stpod"]


# =============================================================================
# Result Container
# =============================================================================


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    case: str
    method: str
    case_type: str  # 'analytical' or 'experimental'
    success: bool
    time_load: float = 0.0
    time_compute: float = 0.0
    time_total: float = 0.0
    n_snapshots: int = 0
    n_spatial: int = 0
    n_modes: int = 0
    error: str = ""
    figures: list = field(default_factory=list)


# =============================================================================
# Data Loader Helper
# =============================================================================


def make_analytical_loader(data: dict) -> Callable:
    """Create a data loader function for analytical data."""

    def loader(file_path):
        return data

    return loader


# =============================================================================
# Benchmark Runners
# =============================================================================


def run_method(
    case_name: str,
    case_type: str,
    config: dict,
    method: str,
    data: Optional[dict] = None,
    generate_plots: bool = True,
) -> BenchmarkResult:
    """Run a single method on a case."""
    result = BenchmarkResult(
        case=case_name, method=method, case_type=case_type, success=False
    )

    # Setup directories
    results_dir = RESULTS_DIR / case_name / method
    figures_dir = FIGURES_DIR / case_name / method
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Determine file path and loader
    if case_type == "analytical":
        file_path = case_name  # Dummy path for analytical
        data_loader = make_analytical_loader(data)
    else:
        file_path = str(DATA_DIR / config["file"])
        data_loader = None

    try:
        t0 = time.perf_counter()

        # Create analyzer based on method
        if method == "pod":
            analyzer = PODAnalyzer(
                file_path=file_path,
                results_dir=str(results_dir),
                figures_dir=str(figures_dir),
                n_modes_save=10,
                spatial_weight_type=config["weight_type"],
                data_loader=data_loader,
            )
            analyzer.load_and_preprocess()
            result.time_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            analyzer.perform_pod()
            result.time_compute = time.perf_counter() - t1
            result.n_modes = len(analyzer.eigenvalues)
            analyzer.save_results()

            if generate_plots and result.n_modes > 0:
                _safe_plot(analyzer.plot_eigenvalues, result)
                _safe_plot(
                    lambda: analyzer.plot_modes(plot_n_modes=min(4, result.n_modes)),
                    result,
                )
                _safe_plot(
                    lambda: analyzer.plot_time_coefficients(
                        n_coeffs_to_plot=min(4, result.n_modes)
                    ),
                    result,
                )
                _safe_plot(analyzer.plot_cumulative_energy, result)

        elif method == "dmd":
            analyzer = DMDAnalyzer(
                file_path=file_path,
                results_dir=str(results_dir),
                figures_dir=str(figures_dir),
                n_modes_save=10,
                spatial_weight_type=config["weight_type"],
                data_loader=data_loader,
            )
            analyzer.load_and_preprocess()
            result.time_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            analyzer.perform_dmd()
            result.time_compute = time.perf_counter() - t1
            result.n_modes = len(analyzer.eigenvalues)
            analyzer.save_results()

            if generate_plots and result.n_modes > 0:
                _safe_plot(analyzer.plot_eigenvalues, result)
                _safe_plot(
                    lambda: analyzer.plot_modes(plot_n_modes=min(4, result.n_modes)),
                    result,
                )
                _safe_plot(
                    lambda: analyzer.plot_time_coefficients(
                        n_coeffs_to_plot=min(4, result.n_modes)
                    ),
                    result,
                )
                _safe_plot(analyzer.plot_cumulative_energy, result)

        elif method == "spod":
            analyzer = SPODAnalyzer(
                file_path=file_path,
                nfft=config["nfft"],
                overlap=0.5,
                results_dir=str(results_dir),
                figures_dir=str(figures_dir),
                spatial_weight_type=config["weight_type"],
                data_loader=data_loader,
            )
            analyzer.load_and_preprocess()
            result.time_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            analyzer.compute_fft_blocks()
            analyzer.perform_spod()
            result.time_compute = time.perf_counter() - t1
            result.n_modes = len(analyzer.freq) if hasattr(analyzer, "freq") else 0
            analyzer.save_results()

            if generate_plots:
                _safe_plot(analyzer.plot_eigenvalues_v2, result)
                _safe_plot(analyzer.plot_cumulative_energy, result)

        elif method == "bsmd":
            analyzer = BSMDAnalyzer(
                file_path=file_path,
                nfft=config["nfft"],
                overlap=0.5,
                results_dir=str(results_dir),
                figures_dir=str(figures_dir),
                spatial_weight_type=config["weight_type"],
                data_loader=data_loader,
            )
            analyzer.load_and_preprocess()
            result.time_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            analyzer.compute_fft_blocks()
            analyzer.perform_bsmd()
            result.time_compute = time.perf_counter() - t1
            result.n_modes = len(analyzer.triads) if hasattr(analyzer, "triads") else 0
            analyzer.save_results()

            if generate_plots:
                _safe_plot(analyzer.plot_energy_map, result)
                _safe_plot(
                    lambda: analyzer.plot_modes(triad_indices=[0, 1], plot_n_modes=2),
                    result,
                )

        elif method == "stpod":
            analyzer = STPODAnalyzer(
                file_path=file_path,
                embedding_dim=config["embedding_dim"],
                n_modes_save=10,
                results_dir=str(results_dir),
                figures_dir=str(figures_dir),
                spatial_weight_type=config["weight_type"],
                data_loader=data_loader,
            )
            analyzer.load_and_preprocess()
            result.time_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            analyzer.perform_stpod()
            result.time_compute = time.perf_counter() - t1
            result.n_modes = (
                len(analyzer.eigenvalues) if len(analyzer.eigenvalues) > 0 else 0
            )
            analyzer.save_results()

            if generate_plots and result.n_modes > 0:
                _safe_plot(analyzer.plot_eigenvalues, result)
                _safe_plot(
                    lambda: analyzer.plot_modes(plot_n_modes=min(4, result.n_modes)),
                    result,
                )
                _safe_plot(
                    lambda: analyzer.plot_time_coefficients(
                        n_coeffs=min(4, result.n_modes)
                    ),
                    result,
                )
                _safe_plot(analyzer.plot_cumulative_energy, result)

        # Extract data info
        if hasattr(analyzer, "data"):
            result.n_snapshots = analyzer.data.get("Ns", 0)
            result.n_spatial = (
                analyzer.data["q"].shape[1] if "q" in analyzer.data else 0
            )

        result.time_total = time.perf_counter() - t0
        result.success = True

    except Exception as e:
        result.error = str(e)
        traceback.print_exc()

    plt.close("all")
    return result


def _safe_plot(plot_func: Callable, result: BenchmarkResult) -> None:
    """Safely execute a plot function, catching errors."""
    try:
        plot_func()
        result.figures.append("generated")
    except Exception as e:
        print(f"    Warning: Plot failed: {e}")


# =============================================================================
# Main Runner
# =============================================================================


def run_benchmarks(
    cases: list,
    methods: list,
    case_configs: dict,
    case_type: str,
    generate_plots: bool = True,
    quick: bool = False,
) -> list:
    """Run benchmarks for a set of cases."""
    results = []

    for case_name in cases:
        if case_name not in case_configs:
            print(f"Warning: Unknown case '{case_name}', skipping.")
            continue

        config = case_configs[case_name]

        # Generate or load data
        if case_type == "analytical":
            print(f"\n{'=' * 60}")
            print(f"  Generating: {case_name}")
            print(f"  {config['description']}")
            print(f"{'=' * 60}")
            data = config["generator"](quick=quick)
        else:
            file_path = DATA_DIR / config["file"]
            if not file_path.exists():
                print(f"Warning: Data file not found: {file_path}, skipping.")
                continue
            print(f"\n{'=' * 60}")
            print(f"  Case: {case_name}")
            print(f"  {config['description']}")
            print(f"{'=' * 60}")
            data = None

        for method in methods:
            print(f"\n  Running {method.upper()}...")
            result = run_method(
                case_name=case_name,
                case_type=case_type,
                config=config,
                method=method,
                data=data,
                generate_plots=generate_plots,
            )
            results.append(result)
            _print_result(result)

    return results


def _print_result(result: BenchmarkResult) -> None:
    """Print a single benchmark result."""
    status = "✓" if result.success else "✗"
    if result.success:
        print(
            f"    {result.method.upper():6s}: "
            f"{result.time_load:5.2f}s load, "
            f"{result.time_compute:6.2f}s compute, "
            f"{result.n_modes:3d} modes {status}"
        )
    else:
        print(f"    {result.method.upper():6s}: FAILED - {result.error[:50]} {status}")


def print_summary(results: list) -> None:
    """Print summary table."""
    # Group by case type
    analytical = [r for r in results if r.case_type == "analytical"]
    experimental = [r for r in results if r.case_type == "experimental"]

    for group_name, group in [("ANALYTICAL", analytical), ("EXPERIMENTAL", experimental)]:
        if not group:
            continue

        cases = sorted(set(r.case for r in group))
        methods = ALL_METHODS

        print(f"\n{'=' * 70}")
        print(f"  {group_name} TIMING SUMMARY (compute time in seconds)")
        print(f"{'=' * 70}")

        header = f"| {'Case':<18} |"
        for m in methods:
            header += f" {m.upper():>7s} |"
        print(header)
        print("|" + "-" * 19 + "|" + ("|" + "-" * 9) * len(methods) + "|")

        for case in cases:
            row = f"| {case:<18} |"
            for m in methods:
                match = [r for r in group if r.case == case and r.method == m]
                if match and match[0].success:
                    row += f" {match[0].time_compute:7.2f} |"
                else:
                    row += "     N/A |"
            print(row)

        print("=" * 70)


def save_results(results: list, output_path: str) -> None:
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="pyModal Unified Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run all benchmarks
  %(prog)s --analytical                 # Analytical cases only
  %(prog)s --experimental               # Experimental data only
  %(prog)s --cases cavity,jet_small     # Specific cases
  %(prog)s --methods pod,dmd            # Specific methods
  %(prog)s --quick                      # Quick mode (smaller grids)
  %(prog)s --no-plots                   # Skip figure generation
        """,
    )
    parser.add_argument(
        "--analytical", action="store_true", help="Run analytical cases only"
    )
    parser.add_argument(
        "--experimental", action="store_true", help="Run experimental cases only"
    )
    parser.add_argument(
        "--cases", type=str, help="Comma-separated case names (e.g., cavity,jet_small)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pod,dmd,spod",
        help="Comma-separated methods (default: pod,dmd,spod)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: smaller grids for analytical, subset cases for experimental"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip figure generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    # Determine methods and validate
    methods = [m.strip().lower() for m in args.methods.split(",")]
    invalid_methods = [m for m in methods if m not in ALL_METHODS]
    if invalid_methods:
        print(f"Warning: Unknown methods ignored: {invalid_methods}")
        print(f"Valid methods: {ALL_METHODS}")
        methods = [m for m in methods if m in ALL_METHODS]
    if not methods:
        print("Error: No valid methods specified.")
        return

    # Determine which cases to run
    run_analytical = not args.experimental or args.analytical
    run_experimental = not args.analytical or args.experimental

    # Handle --cases override
    if args.cases:
        specified_cases = [c.strip() for c in args.cases.split(",")]
        analytical_cases = [c for c in specified_cases if c in ANALYTICAL_CASES]
        experimental_cases = [c for c in specified_cases if c in EXPERIMENTAL_CASES]
        run_analytical = bool(analytical_cases)
        run_experimental = bool(experimental_cases)
    else:
        if args.quick:
            analytical_cases = QUICK_ANALYTICAL if run_analytical else []
            experimental_cases = QUICK_EXPERIMENTAL if run_experimental else []
        else:
            analytical_cases = list(ANALYTICAL_CASES.keys()) if run_analytical else []
            experimental_cases = (
                list(EXPERIMENTAL_CASES.keys()) if run_experimental else []
            )

    # Print header
    print("=" * 70)
    print("  pyModal Unified Benchmark Suite")
    print("=" * 70)
    if run_analytical:
        print(f"  Analytical cases: {', '.join(analytical_cases)}")
    if run_experimental:
        print(f"  Experimental cases: {', '.join(experimental_cases)}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Plots: {'Yes' if not args.no_plots else 'No'}")
    print(f"  Quick mode: {'Yes' if args.quick else 'No'}")
    print("=" * 70)

    all_results = []

    # Run analytical benchmarks
    if run_analytical and analytical_cases:
        print("\n" + "=" * 70)
        print("  ANALYTICAL BENCHMARKS")
        print("=" * 70)
        results = run_benchmarks(
            cases=analytical_cases,
            methods=methods,
            case_configs=ANALYTICAL_CASES,
            case_type="analytical",
            generate_plots=not args.no_plots,
            quick=args.quick,
        )
        all_results.extend(results)

    # Run experimental benchmarks
    if run_experimental and experimental_cases:
        print("\n" + "=" * 70)
        print("  EXPERIMENTAL BENCHMARKS")
        print("=" * 70)
        results = run_benchmarks(
            cases=experimental_cases,
            methods=methods,
            case_configs=EXPERIMENTAL_CASES,
            case_type="experimental",
            generate_plots=not args.no_plots,
            quick=args.quick,
        )
        all_results.extend(results)

    # Summary
    print_summary(all_results)

    # Stats
    successful = sum(1 for r in all_results if r.success)
    failed = sum(1 for r in all_results if not r.success)
    total_figures = sum(len(r.figures) for r in all_results)

    print(f"\n  Total: {len(all_results)} benchmarks")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if not args.no_plots:
        print(f"  Figures saved to: {FIGURES_DIR}")

    # Save results
    save_results(all_results, args.output)

    # List output directories
    print(f"\nResults organized in:")
    print(f"  {RESULTS_DIR}/{{case}}/{{method}}/")
    print(f"  {FIGURES_DIR}/{{case}}/{{method}}/")


if __name__ == "__main__":
    main()
