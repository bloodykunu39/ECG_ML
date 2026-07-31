# Discussion 4.2: Sensitivity to lead-to-polynomial assignment, Data Type 2

**Paper section:** 4.2 (see `../../../draft.pdf`)

Same lead-permutation robustness check as above, Data Type 2.

## Contents

- `main_2d_leg_typ2_permutation.ipynb`

- `EXPERIMENT_2d_leg_typ2_smallcnn_randomseed_permutation/` — per-seed logs (`*_seed40.txt` … `*_seed44.txt`), learning curves, confusion matrix, ROC/PR plots, and a `*_multiseed_summary.csv`

> This experiment also reads the **default-order (Permutation 0) results** from `EXPERIMENT_2d_leg_typ2_smallcnn_randomseed/` for comparison — that folder is a symlink back to its real home in [`../3.2.1_OPI_Legendre_Type2_CNN100/`](../3.2.1_OPI_Legendre_Type2_CNN100/) (it isn't duplicated).

## Dependencies (symlinks, not copies)

- `dataloader.py`, `dataseperation.py`, `encoding.py`, `model_cnn.py`, `model_nn.py`, `resnet1d.py`, `vanilla_transformer_ecg.py`, `plots.py`, `smoothening.py`, `seed_utils.py` → real files in [`../../src/`](../../src/)
- `data_prep/` → real folder in [`../../data_preparation/data_prep/`](../../data_preparation/data_prep/)
- `ML/` → real folder in [`../../data_preparation/ML/`](../../data_preparation/ML/)

## Running

Open the notebook from this folder and run top to bottom. It captures its own directory as
`default_dir` in an early cell and `os.chdir()`s back to it between steps, so it must stay
next to its symlinked dependencies above (don't move the `.ipynb` out of this folder on its own).
Seeds `{40,41,42,43,44}` (see `src/seed_utils.py`) are looped automatically inside the notebook.
