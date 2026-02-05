#!/usr/bin/env python3
"""Command-line interface for pyModal.

Usage:
    pymodal pod --data ./data/file.mat
    pymodal spod --compute --plot
    pymodal --all --data ./data/file.mat
"""

from __future__ import annotations

import argparse
import sys


def main():
    """Main entry point for pymodal CLI."""
    parser = argparse.ArgumentParser(
        description="pyModal — modal decompositions in pure Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    pymodal pod --data ./data/file.mat
    pymodal spod --nfft 256 --overlap 0.5
    pymodal --all --data ./data/file.mat
        """,
    )
    parser.add_argument("method", nargs="?", choices=["pod", "dmd", "spod", "bsmd", "stpod", "all"],
                        default="all", help="Analysis method to run")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--prep", action="store_true", help="Preprocess data only")
    parser.add_argument("--compute", action="store_true", help="Compute analysis only")
    parser.add_argument("--plot", action="store_true", help="Generate plots only")
    parser.add_argument("--nfft", type=int, default=256, help="FFT block size (SPOD/BSMD)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Block overlap (SPOD/BSMD)")
    parser.add_argument("--n-modes", type=int, default=10, help="Number of modes to save")
    parser.add_argument("--embedding-dim", type=int, default=10,
                        help="Time delay embedding dimension (ST-POD)")

    args = parser.parse_args()

    methods = ["pod", "dmd", "spod", "bsmd", "stpod"] if args.method == "all" else [args.method]

    for method in methods:
        print(f"\n{'='*60}")
        print(f" Running {method.upper()}")
        print(f"{'='*60}\n")

        if method == "pod":
            from pymodal.pod import PODAnalyzer
            analyzer = PODAnalyzer(
                file_path=args.data or "data.mat",
                n_modes_save=args.n_modes,
            )
            if not args.compute and not args.plot:
                analyzer.run_analysis()
            elif args.compute:
                analyzer.load_and_preprocess()
                analyzer.perform_pod()
                analyzer.save_results()
            elif args.plot:
                analyzer.load_results()
                analyzer.plot_eigenvalues()
                analyzer.plot_modes()

        elif method == "dmd":
            from pymodal.dmd import DMDAnalyzer
            analyzer = DMDAnalyzer(
                file_path=args.data or "data.mat",
                n_modes_save=args.n_modes,
            )
            if not args.compute and not args.plot:
                analyzer.load_and_preprocess()
                analyzer.perform_dmd()
                analyzer.save_results()
            elif args.compute:
                analyzer.load_and_preprocess()
                analyzer.perform_dmd()
                analyzer.save_results()

        elif method == "spod":
            from pymodal.spod import SPODAnalyzer
            analyzer = SPODAnalyzer(
                file_path=args.data or "data.mat",
                nfft=args.nfft,
                overlap=args.overlap,
            )
            if not args.compute and not args.plot:
                analyzer.run()
                analyzer.perform_spod()
            elif args.compute:
                analyzer.run()
                analyzer.perform_spod()

        elif method == "bsmd":
            from pymodal.bmsd import BSMDAnalyzer
            analyzer = BSMDAnalyzer(
                file_path=args.data or "data.mat",
                nfft=args.nfft,
                overlap=args.overlap,
            )
            if not args.compute and not args.plot:
                analyzer.run()

        elif method == "stpod":
            from pymodal.stpod import STPODAnalyzer
            analyzer = STPODAnalyzer(
                file_path=args.data or "data.mat",
                embedding_dim=args.embedding_dim,
                n_modes_save=args.n_modes,
            )
            if not args.compute and not args.plot:
                analyzer.run_analysis()
            elif args.compute:
                analyzer.load_and_preprocess()
                analyzer.perform_stpod()
                analyzer.save_results()
            elif args.plot:
                analyzer.load_results()
                analyzer.plot_eigenvalues()
                analyzer.plot_modes()

    print("\nDone!")


if __name__ == "__main__":
    main()
