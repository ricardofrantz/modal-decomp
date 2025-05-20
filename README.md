# tri-modal-decomp

A **pure-Python** collection of simple scripts (no MPI, minimal dependencies) for extracting coherent flow structures via three complementary modal decompositions:

- **Bispectral Mode Decomposition (BSMD)**  
  Implements Schmidt (2020) to uncover third-order, phase-coupled structures in turbulent data.  
- **Spectral Proper Orthogonal Decomposition (SPOD)**  
  Identifies energy-optimal spatial modes as functions of frequency, revealing dominant oscillatory patterns.  
- **Time-Optimal Dependent Modes (TOD)**  
  Constructs temporally localized basis functions that minimize reconstruction error for transient events.

---

## 🚀 Key Features

- **Unified API**  
  One simple interface for BSMD, SPOD, and TOD workflows.  
- **Lightweight & Readable**  
  Pure-Python scripts—easy to inspect, modify, and extend.  
- **No MPI Required**  
  Runs out of the box on a single machine.  
- **Flexible I/O**  
  Read/write HDF5, NetCDF, MATLAB `.mat`, or raw NumPy arrays.  
- **Built-in Visualization**  
  Quick plotting of mode shapes, power spectra, and bispectral maps.

---

## 💾 Getting the Code

```bash
git clone https://github.com/ricardofrantz/tri-modal-decomp.git
cd tri-modal-decomp
