# pyModal Documentation

## Development Guidelines

This file provides coding and contribution guidelines for the **pyModal** repository. Follow these directions whenever you modify or add files.

### Project Structure

```
pyModal/
├── src/pymodal/          # Main package
│   ├── pod.py            # Proper Orthogonal Decomposition
│   ├── dmd.py            # Dynamic Mode Decomposition
│   ├── spod.py           # Spectral POD
│   ├── bsmd.py           # Bispectral Mode Decomposition
│   ├── cli.py            # Command-line runner
│   ├── core/             # Shared utilities
│   │   ├── base.py       # BaseAnalyzer class
│   │   ├── config.py     # Configuration settings
│   │   ├── io.py         # Data loaders
│   │   └── parallel.py   # Parallelization helpers
│   └── fft/              # FFT backends
├── examples/             # Benchmark examples
├── tests/                # Unit tests
└── docs/                 # Documentation
```

### Development Workflow

1. Edit Python files using **PEP 8** style with a maximum line length of **120** characters.
2. Prefer single quotes for strings.
3. Run `ruff format` to format code and sort imports.
4. Add descriptive docstrings and follow `snake_case` for functions and `PascalCase` for classes.
5. Keep platform independence in mind—use `pathlib.Path` for paths.

### Testing

Run the unit tests before each commit:

```bash
pytest
```

All tests must pass. The project has minimal dependencies (`h5py`, `matplotlib`, `numpy`, `scipy`, and `tqdm`) and targets **Python 3.14**.

### Running the Scripts

The analysis scripts support staged execution with the flags `--prep`, `--compute`, and `--plot`. OMP-based parallelism is controlled via `OMP_NUM_THREADS`. You can check the detected optimizations with:

```bash
python -c "from pymodal.core.parallel import print_optimization_status; print_optimization_status()"
```

### Commit Messages

Write clear, concise commit messages that describe the motivation for the change. Small, focused commits are preferred.

---

## POD — Proper Orthogonal Decomposition

Data-based techniques only need the flow field data obtained from numerical simulation and do not require knowledge of the governing dynamics. In particular, we consider the proper orthogonal decomposition (POD), which can extract modal contents from a collection of snapshot data.

### Key Concepts

- **Snapshots**: Flow field data collected at an instance in time, formatted into column vectors.
- **Modes ϕᵢ(x)**: Optimally capture the kinetic energy of the unsteady flow field.
- **Eigenvalues λᵢ**: Represent the amount of kinetic energy held by each mode.

The POD analysis finds the best set of spatial modes to extract as much kinetic energy as possible in the flow field over time. These POD modes are orthogonal to each other ensuring the optimality of extracting kinetic energy by each individual mode.

### Snapshot POD Method

The snapshot-based method enables decomposition in a computationally tractable manner when the dimension of an individual snapshot is much larger than the total number of snapshots.

**Preprocessing**: Subtract the mean from all snapshots to focus on modal structures associated with fluctuations.

### Interpreting Results

- Spatial POD modes ϕᵢ(x) capture regions where fluctuations appear in the flow
- For periodic flows, spatial modes appear in pairs (advective physics with oscillator-type dynamics)
- The dominant modes reveal the dominant energetic spatial structures
- POD modes are orthogonal: ⟨ϕᵢ, ϕⱼ⟩ = δᵢⱼ

### Limitations

Modes extracted from the input flow field data are optimally determined for the provided data and may not generalize to perturbed flows. For better coverage:
- Repeat POD analysis with perturbed flow field data
- Consider Balanced POD analysis (requires adjoint simulation)

---

## DMD — Dynamic Mode Decomposition

DMD analysis extracts spatial modes with associated frequencies and growth/decay rates from snapshot data. Unlike POD, mean subtraction is not necessary.

### Key Outputs

- **Static mode**: DMD eigenvalue λ = 1 corresponds to the mean flow
- **Oscillatory modes**: Appear in complex-conjugate pairs
- **Eigenvalues**: Encode frequency fᵢ = ∠λᵢ/(2πδt) and growth rate gᵢ = log|λᵢ|/δt

### DMD vs POD

| Aspect | POD | DMD |
|--------|-----|-----|
| Modes | Real-valued | Complex-valued |
| Orthogonality | Always orthogonal | Not necessarily orthogonal |
| Temporal info | Via time coefficients | Via eigenvalues (single frequency) |
| Mean subtraction | Required | Not required |

**For periodic flows**: POD and DMD yield similar spatial modes. POD modes 1,2 correspond to Re/Im parts of DMD mode 1.

### Magnitude/Phase Representation

- **Magnitude plots**: Reveal active regions of each mode
- **Phase plots**: Display relative phase between spatial regions

### Practical Considerations

1. **Data collection matters**: Don't "throw" DMD at full datasets—separate distinct flow regimes (linear dynamics, transient, limit cycle)
2. **Noise sensitivity**: DMD exhibits biased results with measurement uncertainties. Use noise-robust variants for experimental data.

---

## Mathematical Details

### POD Formulation

Given snapshot matrix X ∈ ℝ^(N×M), POD solves:

```
X = Φ Σ Ψᵀ  (SVD)
```

Where Φ contains spatial modes, Σ contains singular values (√eigenvalues), and Ψ contains temporal coefficients.

### DMD Formulation

For snapshots X₁ and X₂ (shifted by one timestep), DMD finds the best-fit linear operator A:

```
X₂ ≈ A X₁
```

DMD modes are eigenvectors of A, eigenvalues encode dynamics.

### SPOD Formulation

For stationary data, SPOD solves eigenproblems of the cross-spectral density matrix at each frequency:

```
Mᵢ = Xᵢᴴ W Xᵢ
```

Where Xᵢ are FFT blocks at frequency fᵢ and W is the spatial weight matrix.

### BSMD Formulation

For triads (p₁, p₂, p₃) with f_{p₁} + f_{p₂} = f_{p₃}:

```
C = A† W B,  C a = λ a
```

Reveals third-order phase-coupled interactions driving nonlinear energy transfer.
