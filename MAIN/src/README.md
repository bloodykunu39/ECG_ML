# `src/` — Core Modules

Shared Python modules used by every experiment notebook under `../run_experiments/`,
`../invertibility/`, and `../analysis_and_utilities/`. These are the **only real copies** —
every place that needs them holds a symlink back here, so editing a file in `src/` updates
it everywhere at once.

| Module | Purpose |
|---|---|
| `dataloader.py` | `MyCustomDataset` — PyTorch `Dataset` wrapper used by every training notebook. |
| `dataseperation.py` | Splits/filters raw ECG records by disease code (`disease_codes`, `datasepration_single`) into the Type 1 / Type 2 datasets described in `data_preparation/`. |
| `encoding.py` | The orthogonal-polynomial imaging (OPI) encoding itself: `normalize_matrix`, `superposition`, `inverse_superposition` — maps 12 one-dimensional leads onto a single 2D image via Legendre/Chebyshev polynomials (paper §2.1.3). |
| `smoothening.py` | `coarsegrain(img, cg=50)` — downsamples the 5000×5000 encoded image to 100×100 / 50×50 via coarse-graining (paper §2.1.3). |
| `model_cnn.py` | 2D CNN architectures (`SmallCNN`, `SmallCNN50`, `LargeCNN`) used on the OPI-encoded images (paper §2.2.2). |
| `model_nn.py` | `Model` — the single-lead feed-forward network baseline (paper §2.2.1). |
| `resnet1d.py` | `BenchmarkResidualBlock1D` and related building blocks backing the SelfEEG ResNet1D baseline (paper §2.2.5). |
| `vanilla_transformer_ecg.py` | `PositionalEncoding` and the Vanilla Transformer model (paper §2.2.3). Imports `encoding.py` internally. |
| `plots.py` | All evaluation/plotting: `evaluate_all`, `evaluate_multiseed_plots`, `plot_learning_curves_multiseed`, `summarize_seed_results`, confusion-matrix/ROC/PR figure generation, `accuracy_and_validation_plots`. Imports `dataloader.py` internally. |
| `seed_utils.py` | `SEEDS = [40, 41, 42, 43, 44]` and `set_seed()` — the fixed 5-seed reproducibility protocol (paper §2.3.1) used by every experiment for its multi-seed mean ± std results. |

## Note on `encoding2.py`

The old `MAIN/README.md` documented an `encoding2.py` ("alternative/updated encoding
implementations") and one notebook (`data_preparation/ML/smooth_using_normlised_legendre.ipynb`)
imports it — but the file does not exist anywhere in this repository. This is a pre-existing gap,
not something introduced by this reorganization; that one notebook was already unrunnable as-is.
