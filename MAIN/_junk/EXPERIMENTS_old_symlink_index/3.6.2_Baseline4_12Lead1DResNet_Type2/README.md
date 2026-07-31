# Baseline 4: 12-lead raw ECG, SelfEEG ResNet1D, Data Type 2

**Paper section:** 3.6.2 (see `draft.pdf`)

SelfEEG ResNet1D on raw 12x5000 ECG waveform, Data Type 2.

## Contents (symlinks to the real files — run from their original location in `MAIN/`)

- Run notebook(s): `main_1d_resnet_typ2_selfeeg.ipynb`
- Results / plots / summary CSVs: `results`

> Note: these are symlinks for navigation only. The notebooks rely on relative
> imports (`dataloader.py`, `model_cnn.py`, `plots.py`, etc.) and relative data
> paths that assume the working directory is `MAIN/`. Open/run the notebook
> from its original path in `MAIN/`, not from inside this folder.
