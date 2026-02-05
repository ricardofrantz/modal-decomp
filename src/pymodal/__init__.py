"""pyModal — modal decompositions in pure Python.

A lightweight, zero-MPI toolkit for extracting coherent structures
from spatiotemporal data.

Example usage:
    from pymodal import PODAnalyzer, DMDAnalyzer, SPODAnalyzer

    pod = PODAnalyzer(file_path="data.mat", n_modes_save=10)
    pod.run_analysis()
"""

from pymodal.pod import PODAnalyzer
from pymodal.dmd import DMDAnalyzer
from pymodal.spod import SPODAnalyzer
from pymodal.bmsd import BSMDAnalyzer
from pymodal.stpod import STPODAnalyzer

__version__ = "0.1.0"
__all__ = ["PODAnalyzer", "DMDAnalyzer", "SPODAnalyzer", "BSMDAnalyzer", "STPODAnalyzer"]
