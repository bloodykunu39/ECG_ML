# Baseline 5: Single-lead raw ECG, SelfEEG ResNet1D, Data Type 1

**Paper section:** 3.7.1 (see `draft.pdf`)

SelfEEG ResNet1D trained on Lead 1 only (1x5000), Data Type 1.

## Contents (symlinks to the real files — run from their original location in `MAIN/`)

- Run notebook(s): `main_1d_resnet_typ1_selfeeg_singlelead1.ipynb`
- Results / plots / summary CSVs: `results`

> Note: these are symlinks for navigation only. The notebooks rely on relative
> imports (`dataloader.py`, `model_cnn.py`, `plots.py`, etc.) and relative data
> paths that assume the working directory is `MAIN/`. Open/run the notebook
> from its original path in `MAIN/`, not from inside this folder.
