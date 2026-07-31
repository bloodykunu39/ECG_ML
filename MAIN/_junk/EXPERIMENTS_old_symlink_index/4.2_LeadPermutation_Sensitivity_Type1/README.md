# Discussion 4.2: Sensitivity to lead-to-polynomial assignment, Data Type 1

**Paper section:** 4.2 (see `draft.pdf`)

Legendre CNN (100x100) retrained under 4 random lead-to-polynomial permutations vs. default assignment, to test robustness of OPI to lead ordering. Data Type 1.

## Contents (symlinks to the real files — run from their original location in `MAIN/`)

- Run notebook(s): `main_2d_leg_typ1_permutation.ipynb`
- Results / plots / summary CSVs: `results_EXPERIMENT_2d_leg_typ1_smallcnn_randomseed`, `results_EXPERIMENT_2d_leg_typ1_smallcnn_randomseed_permutation`

> Note: these are symlinks for navigation only. The notebooks rely on relative
> imports (`dataloader.py`, `model_cnn.py`, `plots.py`, etc.) and relative data
> paths that assume the working directory is `MAIN/`. Open/run the notebook
> from its original path in `MAIN/`, not from inside this folder.
