# Invertibility of the OPI encoding

**Paper section:** 3.9 (see `draft.pdf`)

Quantifies reconstruction of original ECG leads from the superposed image prior to smoothing (MSE, MAE, relative L2 error), for both Data Type 1 and Type 2. Notebooks and result CSVs/plots already co-located in invertibility/.

## Contents (symlinks to the real files — run from their original location in `MAIN/`)

- Run notebook(s): `invertibility_typ1.ipynb`, `invertibility_typ2.ipynb`
- Results / plots / summary CSVs: `results`

> Note: these are symlinks for navigation only. The notebooks rely on relative
> imports (`dataloader.py`, `model_cnn.py`, `plots.py`, etc.) and relative data
> paths that assume the working directory is `MAIN/`. Open/run the notebook
> from its original path in `MAIN/`, not from inside this folder.
