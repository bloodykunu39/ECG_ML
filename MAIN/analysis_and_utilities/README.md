# `analysis_and_utilities/` — Cross-Experiment Tools

Notebooks that operate *across* multiple experiments rather than running one of their own,
so they don't belong under `run_experiments/`.

- **`analysis.ipynb`** — Diagnostic dashboard that globs every `EXPERIMENT_*_randomseed`
  folder it can see, compiles metrics across all of them, and plots aggregate comparison
  charts. Supports the cross-model comparisons in the paper's Discussion §4.1. Its glob
  needs every experiment's results folder to be visible here, so this folder holds a symlink
  to each of the 20 `EXPERIMENT_*` folders now living under `../run_experiments/*/` (see
  `ls EXPERIMENT_*` — none of that data is duplicated, only linked).
- **`12leadstack.ipynb`** — Exploratory notebook evaluating a stacking architecture across
  multiple leads; not one of the paper's reported experiments.
- **`Extractor.ipynb`** — Dev tool for pulling cells/outputs out of other notebooks into a
  results folder. Some of its `NOTEBOOK_MAP` entries reference notebook filenames
  (`main_1d_resnet_typ1.ipynb`, `main_1d_resnet_typ2.ipynb`) that don't exist in this
  repository (likely superseded by the `_selfeeg` variants) — this was already the case
  before the reorganization and was left unchanged.

## Dependencies (symlinks, not copies)

`dataloader.py`, `dataseperation.py`, `encoding.py`, `model_cnn.py`, `model_nn.py`,
`resnet1d.py`, `vanilla_transformer_ecg.py`, `plots.py`, `smoothening.py`, `seed_utils.py`
→ real files in [`../src/`](../src/). `data_prep/`, `ML/` → real folders in
[`../data_preparation/`](../data_preparation/). `EXPERIMENT_*` → real folders under
[`../run_experiments/`](../run_experiments/).
