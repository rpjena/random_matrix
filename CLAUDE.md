# CLAUDE.md

## Project Overview

This is a **research/learning project** exploring **Random Matrix Theory** and related quantitative finance methods. It is inspired by the random matrix theory book by JPB and MP, aiming to decipher the book through implementations and exercises.

All code lives in Jupyter notebooks (`.ipynb`) designed to run in **Google Colab**. There is no package structure, build system, or CI/CD pipeline.

## Repository Structure

```
random_matrix/
├── CLAUDE.md                                # This file
├── README.md                                # Minimal project description
├── chapter1.ipynb                           # Banded matrices, eigenvalue solving, roots of unity
├── hiddenfactor.ipynb                       # Hidden factor estimation with time-varying betas
├── ipca_myver.ipynb                         # IPCA (Instrumental PCA) implementation - primary
├── ipca_myver2.ipynb                        # IPCA implementation variant
├── ipcanew1.ipynb                           # Alternative IPCA implementation
├── ipcarud.ipynb                            # IPCA variant
├── mctr.ipynb                               # Marginal Contribution to Total Risk (MCTR/ACTR)
├── CUSUM_Change_Detection_in_Python.ipynb   # CUSUM change detection
├── cumsumcode.ipynb                         # Extended CUSUM implementation
├── Untitled3.ipynb                          # Exploratory notebook
├── Untitled4.ipynb                          # Exploratory notebook
└── Untitled5.ipynb                          # Exploratory notebook
```

## Key Topics Covered

- **Random Matrix Theory**: Banded matrices, eigenvalue analysis, spectral methods (`chapter1.ipynb`)
- **Factor Models / IPCA**: Instrumental Principal Component Analysis with alternating least squares (`ipca_myver.ipynb` and variants)
- **Hidden Factor Estimation**: Estimating latent factor returns from time-varying betas (`hiddenfactor.ipynb`)
- **Risk Attribution**: Portfolio marginal/absolute contribution to risk (`mctr.ipynb`)
- **Change Detection**: CUSUM (Cumulative Sum Control Chart) methods (`CUSUM_Change_Detection_in_Python.ipynb`, `cumsumcode.ipynb`)

## Language and Dependencies

**Language**: Python 3 (all code in Jupyter notebooks)

**Required Libraries** (install manually; no `requirements.txt` exists):
- `numpy` - numerical computing, matrix operations
- `pandas` - data manipulation, DataFrames
- `scipy` - scientific computing (`scipy.linalg`, `scipy.sparse.linalg`, `scipy.optimize`)
- `matplotlib` - plotting
- `seaborn` - statistical visualization
- `statsmodels` - statistical modeling

## Development Workflow

### Running Notebooks
All notebooks include a Google Colab badge and are designed to be opened and run in Colab:
1. Open notebook in Google Colab via the badge link at the top of each notebook
2. Run cells sequentially
3. Dependencies are expected to be pre-installed in the Colab environment

### No Build / Test / Lint Commands
- There is no build system, test suite, or linting configuration
- Validation is done manually by inspecting notebook outputs
- There are no CI/CD pipelines or pre-commit hooks

## Code Conventions

- **Docstrings**: NumPy-style docstrings with Parameters/Returns sections (see `hiddenfactor.ipynb`, `ipca_myver.ipynb`)
- **MATLAB-inspired helpers**: Some functions follow MATLAB naming (e.g., `_mldivide`, `_mrdivide` in `ipca_myver.ipynb`)
- **Underscore prefix**: Internal/helper functions use `_` prefix (e.g., `_sign_convention`, `_calc_r2`)
- **Class-based models**: The IPCA implementation uses a class (`IPCA`) with `__init__`, `run_ipca`, `fit` methods
- **Data structures**: Heavy use of `dict(T) of df(NxL)` patterns - dictionaries keyed by time period containing DataFrames
- **No type annotations** in function signatures; types documented in docstrings instead

## Important Notes for AI Assistants

1. **This is a research notebook project** - do not try to add package infrastructure, CI/CD, or testing frameworks unless explicitly asked
2. **Notebooks may contain errors** - some cells have runtime errors in their output (e.g., `ipca_myver.ipynb` has a data type mismatch). This is expected in exploratory work
3. **"Untitled" notebooks** are work-in-progress explorations - treat them as scratch/experimental
4. **Single author project** - all commits are from `rpjena`, commit messages are typically "Created using Colab"
5. **No secrets or credentials** are stored in the repository
6. **All notebooks assume Colab environment** - paths and imports are written for Google Colab
