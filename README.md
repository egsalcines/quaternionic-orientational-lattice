# Curvature localization and finite-size scaling in a discrete quaternionic orientational lattice

This repository contains the numerical simulation code, datasets, and figure-generation outputs associated with the manuscript:

**Curvature localization and finite-size scaling in a discrete quaternionic orientational lattice**

The model studies a discrete orientational lattice with local quaternionic degrees of freedom under fixed boundary-degree constraints. The simulations explore curvature localization, finite-size scaling, energy decomposition, boundary-degree conservation, and morphological transitions.

## Repository contents

- `quaternionic_lattice_simulation.py`: main simulation script.
- `outputs/csv/`: numerical results from the robust sweep.
- `outputs/data/`: compressed morphology fields used for representative configurations and compactness analysis.
- `outputs/figures/`: generated figures used in the manuscript.
- `paper/`: manuscript PDF.

# Installation

Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Quick test run

```bash
python quaternionic_lattice_simulation.py \
  --N-values 16 24 \
  --Q-values 1 2 3 4 \
  --seeds 3 \
  --steps 2000 \
  --adaptive \
  --min-steps 1000 \
  --delta-E-tol 1e-5 \
  --patience 3
```

---

# Main run used for the manuscript

```bash
python quaternionic_lattice_simulation.py \
  --N-values 16 24 32 \
  --Q-values 1 2 3 4 5 6 7 8 9 10 \
  --seeds 8 \
  --steps 9000 \
  --adaptive \
  --min-steps 3000 \
  --delta-E-tol 1e-5 \
  --patience 5
```

---

# Generated outputs

The script automatically creates:

```text
outputs/csv/
outputs/data/
outputs/figures/
```

The main CSV files are:

- `robust_all_runs.csv`: all individual simulations.
- `robust_summary_by_N_Q.csv`: statistics grouped by lattice size and boundary degree.
- `robust_best_by_N_Q.csv`: minimum-energy configuration for each `(N,Q)`.
- `robust_fits_by_N.csv`: power-law fits by lattice size.
- `robust_alpha_extrapolation.csv`: finite-size extrapolation data.
- `robust_run_parameters.csv`: parameters used in the run.

---

# Figures

The script generates:

- energy scaling figure,
- finite-size exponent figure,
- boundary-degree conservation figure,
- morphology panels.

Additional postprocessed figures, such as the curvature compactness and absolute energy decomposition plots, are included in `outputs/figures/`.

---

# Notes

The simulations are computationally intensive for larger lattices and high boundary degree. Running the full manuscript configuration may take a long time depending on hardware.

The reported results were obtained using fixed lattice connectivity. Only the quaternionic orientational degrees of freedom evolve dynamically.
