# `invertibility/` — Paper §3.9

Quantifies how well the original 12 ECG leads can be reconstructed from the superposed OPI
image *before* coarse-graining/smoothing is applied (once resolution is reduced, the encoding
is no longer strictly invertible — see paper §3.9).

Left in place at this level (not moved under `../run_experiments/`) because it's already a
single self-contained unit: notebook + results side by side, and its own `sys.path`/relative
`data_prep` references are one directory level up from here, matching where this folder already
sits relative to `../data_preparation/data_prep`.

## Contents

- `invertibility_typ1.ipynb`, `invertibility_typ2.ipynb` — reconstruction notebooks for Data
  Type 1 / Type 2, computing MSE, MAE, and relative ℓ₂ error (`from encoding import
  normalize_matrix, superposition, inverse_superposition` — imported from `../src/encoding.py`
  via `sys.path.append(os.path.abspath(".."))`, and reads `../data_preparation/data_prep/...`
  via the `data_prep` symlink one level up... see note below).
- `result_of_TYP1_*.csv`, `result_of_typ2_*.csv`, `results_combined.csv` — per-disease and
  combined reconstruction-error tables.
- `histograms_TYP1*.png`, `histograms_TYP2*.png` — error-distribution plots (Figure 53 in the
  paper).

## Path note

These two notebooks read `../data_prep/...` (one level up). Since `data_prep` moved into
`../data_preparation/data_prep/` as part of this reorganization, a `data_prep` symlink was
added directly in `MAIN/` pointing at `data_preparation/data_prep` so the existing `../data_prep`
reference in these notebooks keeps resolving without any edits to their code.
