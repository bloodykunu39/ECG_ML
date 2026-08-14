# Results

This directory holds the outputs of every training run reported in the paper, plus the notebook that aggregates them.

## Contents

- **`analysis.ipynb`** — discovers every `EXPERIMENT_*_randomseed/` folder below it (via `Path.cwd().glob("EXPERIMENT*_randomseed")`), reads each one's summary CSV, and produces cross-experiment ranking/comparison plots (accuracy bar charts, metric heatmaps, seed-wise boxplots).
- **`EXPERIMENT_*_randomseed/`** — one folder per experimental configuration in the paper (encoding × dataset type × architecture, e.g. `EXPERIMENT_2d_leg_typ1_smallcnn_randomseed` = Legendre-encoded CNN, Type 1). Each folder contains, per seed (40–44):
  - `main_*_seed<NN>.txt` — training log for that seed
  - `*_multiseed_cm.png`, `*_multiseed_pr.png`, `*_multiseed_roc.png` — confusion matrix / precision-recall / ROC curves aggregated across seeds
  - `*_val_acc_multiseed.png`, `*_val_loss_multiseed.png` — validation curves across seeds
  - `*_multiseed_summary.csv` — per-seed and mean±std metrics (accuracy, F1, AUC, AP) — this is what `analysis.ipynb` reads

  Two configurations (`EXPERIMENT_2d_leg_typ1_smallcnn_randomseed_permutation/`, `EXPERIMENT_2d_leg_typ2_smallcnn_randomseed_permutation/`) additionally have `perm1`–`perm4` subfolders, one per lead-order permutation, corresponding to the paper's lead-ordering robustness check.

Each `EXPERIMENT_*` folder is produced by the correspondingly-named `main_*.ipynb` notebook at the repo root — see the root [README](../README.md#reproducing-the-experiments) for the full pipeline.

## Note

An earlier version of this documentation (previously `MAIN/README.md`) described each experiment folder as containing `model_weights/`, `experiment_config.json`, and `seed_record.txt`. That doesn't match what's actually produced by the training notebooks (logs + plots + summary CSV, as above) — this README reflects the folders as they actually exist.
