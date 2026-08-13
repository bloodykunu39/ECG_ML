# ECG Classification using High Energy Physics-Inspired ML Techniques

This project classifies 12-lead ECG signals using an orthogonal-polynomial imaging (OPI)
encoding — inspired by partial-wave decomposition in high-energy physics — that maps
multi-lead 1D signals into a single 2D image, then benchmarks CNNs/ResNets/Transformers
trained on that representation against several raw-signal baselines. 

---

## 🧭 Start here

**[`run_experiments/README.md`](./run_experiments/README.md)** indexes every experiment in
the paper's Results (§3) and Discussion (§4.2) by section number, linking each to its real
run notebook and results folder. That's the fastest way to find "the notebook that made
Figure/Table N."

---

## 📂 Folder Layout

```
MAIN/
├── src/                    Core Python modules (the only real copies — see src/README.md)
├── data_preparation/       Raw data → filtered signals → OPI-encoded images (see its README)
│   ├── zenodo_data/           raw PhysioNet extraction
│   ├── data_prep/             disease filtering → Type 1 / Type 2 .npy datasets
│   ├── ML/                    polynomial encoding + coarse-graining → image datasets
│   └── save_data.ipynb        export utility
├── run_experiments/        One real folder per paper experiment (notebook + its results)
│   └── <paper §>_<name>/      e.g. 3.1.1_OPI_Legendre_Type1_CNN100/
├── invertibility/          Paper §3.9 — reconstruction-error analysis (self-contained)
├── analysis_and_utilities/ Cross-experiment tools: analysis.ipynb, Extractor.ipynb, 12leadstack.ipynb
├── exploratory_notebooks/  Early scratch notebooks, not part of the paper (cg_200.ipynb, superpostion_inverse.ipynb)
├── assets/                 Static images/example files, not read by any code
├── data_prep, ML            (compatibility symlinks → data_preparation/*, see note below)
├── _junk/                  Sandbox-generated debris from building this index — safe to delete manually
└── README.md               This file
```

## Why some folders contain symlinks alongside real files

Every `main_*.ipynb` experiment notebook captures its own working directory in a
`default_dir = os.getcwd()` variable early on, `os.chdir()`s back to it between steps, and
loads data / writes results using **bare relative names** (`"data_prep"`, `"ML/data_unq"`,
`"EXPERIMENT_2d_leg_typ1_smallcnn_randomseed"`, `import dataloader`, ...). That only works if
those names exist as real entries in the notebook's own folder.

Rather than edit that logic inside 20+ large notebooks (which would risk changing how the
paper's actual results were produced — the one thing explicitly *not* wanted here), each
notebook's dependencies are made to exist alongside it via symlinks to the single real copy:

- `run_experiments/<experiment>/*.py` → real files in `src/`
- `run_experiments/<experiment>/data_prep`, `.../ML` → real folders in `data_preparation/`
- `run_experiments/<experiment>/EXPERIMENT_...` (permutation experiments only) → the shared
  default-order results, really owned by the corresponding §3.1.1/§3.2.1 folder
- `analysis_and_utilities/` gets the same treatment plus a symlink to every `EXPERIMENT_*`
  folder, since `analysis.ipynb` globs across all of them at once
- `MAIN/data_prep`, `MAIN/ML` — two root-level compatibility symlinks, kept because
  `invertibility/*.ipynb` reference `../data_prep` (one level above `invertibility/`, which is
  still `MAIN/`)

**Nothing is duplicated.** Every symlink resolves to exactly one real file or folder, and no
notebook, module, or dataset was edited — only relocated, with byte-for-byte content verified
against git history during this reorganization.

---

## 🔬 Experiments (paper §3 Results, §4.2 Discussion)

See **[`run_experiments/README.md`](./run_experiments/README.md)** for the full linked index.
Summary:

| # | Paper § | What |
|---|---|---|
| 1–2 | 3.1 | OPI (Legendre), CNN, Data Type 1 — 100×100 and 50×50 |
| 3–4 | 3.2 | OPI (Legendre), CNN, Data Type 2 — 100×100 and 50×50 |
| 5–8 | 3.3 | Baseline 1: 2D ECGResNet — Type 1/2 × pretrained/non-pretrained |
| 9–10 | 3.4 | Baseline 2: Vanilla Transformer — Type 1/2 |
| 11–12 | 3.5 | Baseline 3: Single-lead FFNN — Type 1/2 |
| 13–14 | 3.6 | Baseline 4: 12-lead SelfEEG ResNet1D — Type 1/2 |
| 15–16 | 3.7 | Baseline 5: Single-lead SelfEEG ResNet1D — Type 1/2 |
| 17–18 | 3.8 | Alternative encoding (Chebyshev-CNN) — Type 1/2 |
| — | 3.9 | Invertibility of the OPI encoding — see `invertibility/` |
| 19–20 | 4.2 | Lead-to-polynomial permutation sensitivity — Type 1/2 |

## 🌱 Reproducibility & Seeds

Every experiment above loops over the same 5 fixed seeds, defined once in
[`src/seed_utils.py`](./src/seed_utils.py):

```python
SEEDS = [40, 41, 42, 43, 44]
```

`set_seed()` fixes `random`, `numpy`, and `torch` (CPU + CUDA) RNG state and disables cuDNN
nondeterminism for each run. Every `EXPERIMENT_*` results folder contains one `.txt` training
log per seed (`..._seed40.txt` … `..._seed44.txt`) plus a `..._multiseed_summary.csv` with the
mean ± std metrics reported in the paper's tables, and multiseed learning-curve / confusion
matrix / ROC / PR plots.

---

## 🛠️ Core Modules

See **[`src/README.md`](./src/README.md)** for the full description of each module
(`dataloader.py`, `dataseperation.py`, `encoding.py`, `model_cnn.py`, `model_nn.py`,
`resnet1d.py`, `vanilla_transformer_ecg.py`, `plots.py`, `smoothening.py`, `seed_utils.py`).

## 🗄️ Data Preparation

See **[`data_preparation/README.md`](./data_preparation/README.md)** for the raw → filtered →
encoded pipeline, disease categories/sample sizes, and known pre-existing gaps (e.g. a missing
`encoding2.py` referenced by one notebook, and a few sub-notebooks with hardcoded absolute
paths from the original author's machine that were already unrunnable before this reorg).

## 📈 Analysis & Utilities

See **[`analysis_and_utilities/README.md`](./analysis_and_utilities/README.md)**.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 1.9+
* NumPy, Pandas, scikit-learn, Matplotlib
* `selfeeg` (for the ResNet1D baselines)

### Running an experiment
1. Open `run_experiments/<experiment>/README.md` to confirm you have the right one.
2. Open the `.ipynb` in that same folder in Jupyter and run top to bottom — its symlinked
   dependencies (modules + data) are already in place, and results land back in that folder's
   `EXPERIMENT_*` subfolder.
