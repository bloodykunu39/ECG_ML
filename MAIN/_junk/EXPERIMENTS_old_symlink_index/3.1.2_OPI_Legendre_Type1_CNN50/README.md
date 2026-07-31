# OPI (Legendre) benchmarking, CNN, reduced 50x50, Data Type 1

**Paper section:** 3.1.2 (see `draft.pdf`)

Robustness check: SmallCNN trained on more aggressively downsampled 50x50 Legendre OPI images, Data Type 1.

## Contents (symlinks to the real files — run from their original location in `MAIN/`)

- Run notebook(s): `main_2d_leg_on_50_typ1.ipynb`
- Results / plots / summary CSVs: `results`

> Note: these are symlinks for navigation only. The notebooks rely on relative
> imports (`dataloader.py`, `model_cnn.py`, `plots.py`, etc.) and relative data
> paths that assume the working directory is `MAIN/`. Open/run the notebook
> from its original path in `MAIN/`, not from inside this folder.
