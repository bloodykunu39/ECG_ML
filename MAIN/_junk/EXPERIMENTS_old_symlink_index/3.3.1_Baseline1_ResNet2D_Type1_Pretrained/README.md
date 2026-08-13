# Baseline 1: 2D ECGResNet (ImageNet-pretrained), Data Type 1

**Paper section:** 3.3.1 (see `draft.pdf`)

Standard 2D image-based ResNet baseline trained on the raw lead-by-time ECG matrix (12x5000), pretrained weights, Data Type 1.

## Contents (symlinks to the real files — run from their original location in `MAIN/`)

- Run notebook(s): `main_2d_resnet_typ1.ipynb`
- Results / plots / summary CSVs: `results`

> Note: these are symlinks for navigation only. The notebooks rely on relative
> imports (`dataloader.py`, `model_cnn.py`, `plots.py`, etc.) and relative data
> paths that assume the working directory is `MAIN/`. Open/run the notebook
> from its original path in `MAIN/`, not from inside this folder.
