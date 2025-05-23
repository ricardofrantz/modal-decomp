# modal-decomp

A **pure-Python** collection of simple scripts (no MPI, minimal dependencies) for extracting coherent flow structures via modal decompositions:

- **Spectral Proper Orthogonal Decomposition (SPOD)**  
  Solves the cross-spectral density eigenvalue problem to yield energy-ranked, harmonic spatial modes under the assumption of wide-sense stationarity.  
  _Reference:_ [Towne, Schmidt & Colonius (2018)](https://arxiv.org/abs/1708.04393)

- **Bispectral Mode Decomposition (BSMD)**  
  Extracts third-order phase-coupled spatial modes by diagonalizing an estimated bispectral density tensor, revealing the triadic interactions that drive nonlinear energy transfer.
  _Reference:_ [Nekkanti, Pickering, Schmidt & Colonius (2025)](https://arxiv.org/abs/2502.15091)

- **Space-Time Proper Orthogonal Decomposition (ST-POD)**  
  Generalizes POD to a full space–time framework by solving eigenproblem of the space-time correlation tensor, capturing arbitrary nonstationary and transient dynamics over finite windows.  
  _Reference:_ [Yeung & Schmidt (2025)](https://arxiv.org/abs/2502.09746)

---

## 🚀 Key Features

- **Unified API**  
  One simple interface for the workflows.  
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
