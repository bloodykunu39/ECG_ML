# Baseline 4: 12-lead raw ECG, SelfEEG ResNet1D, Data Type 1

**Paper section:** 3.6.1 (see `../../../draft.pdf`)

Temporal 1D residual CNN (SelfEEG ResNet1D) trained directly on the raw 12x5000 ECG waveform, Data Type 1.

## Contents

- `main_1d_resnet_typ1_selfeeg.ipynb`

- `EXPERIMENT_resnet_1d_selfeeg_typ1_randomseed/` — per-seed logs (`*_seed40.txt` … `*_seed44.txt`), learning curves, confusion matrix, ROC/PR plots, and a `*_multiseed_summary.csv`

## Dependencies (symlinks, not copies)

- `dataloader.py`, `dataseperation.py`, `encoding.py`, `model_cnn.py`, `model_nn.py`, `resnet1d.py`, `vanilla_transformer_ecg.py`, `plots.py`, `smoothening.py`, `seed_utils.py` → real files in [`../../src/`](../../src/)
- `data_prep/` → real folder in [`../../data_preparation/data_prep/`](../../data_preparation/data_prep/)
- `ML/` → real folder in [`../../data_preparation/ML/`](../../data_preparation/ML/)

## Running

Open the notebook from this folder and run top to bottom. It captures its own directory as
`default_dir` in an early cell and `os.chdir()`s back to it between steps, so it must stay
next to its symlinked dependencies above (don't move the `.ipynb` out of this folder on its own).
Seeds `{40,41,42,43,44}` (see `src/seed_utils.py`) are looped automatically inside the notebook.
