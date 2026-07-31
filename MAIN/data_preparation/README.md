# `data_preparation/` — Raw Data → Filtered Signals → OPI-Encoded Images

The three stages of the pipeline described in the paper's §2.1 (Data Processing), grouped
together. Each stage is its own real, untouched folder (internal contents/notebooks were
not modified, only relocated as a unit):

## `zenodo_data/`
Raw 12-lead ECG source data and extraction notebooks (PhysioNet arrhythmia database,
downloaded/staged here). See `zenodo_data/data_analysis_big.ipynb`, `zenedo_single_disease.ipynb`.

## `data_prep/`
Filters the raw records down to the three disease categories used throughout the paper
(paper §2.1.2, Table in `data_prep/README.md`):

| Disease | Code | Type 1 (single-diagnosis) | Type 2 (with comorbidities) |
|---|---|---|---|
| ST | 427084000 | 3000 | 5000 |
| SB | 426177001 | 5000 | 5000 |
| SR | 426783006 | 5000 | 5000 |

`unq_disease_*.npy` = Type 1, `disease_*.npy` = Type 2. Two source files were found corrupted
and excluded (`23 236 JS23074`, `01 019 JS01052`). Produced by `data_prep.ipynb` and
`data_prep_for_only_one_disease.ipynb` — see `data_prep/README.md` for the full write-up.

## `ML/`
Encodes the filtered 1D signals into 2D images via `src/encoding.py` (Legendre / Chebyshev /
Hermite superposition) and `src/smoothening.py` (coarse-graining), producing the datasets each
`run_experiments/` notebook trains on:

| Directory | Resolution | Type | Encoding |
|---|---|---|---|
| `ML/data_unq` | 100×100 | 1 | Legendre |
| `ML/data_4` | 100×100 | 2 | Legendre |
| `ML/data_unq_50` | 50×50 | 1 | Legendre |
| `ML/data_50` | 50×50 | 2 | Legendre |
| `ML/Data_cheb` | 100×100 | 1 | Chebyshev |
| `ML/Data_cheb_typ2` | 100×100 | 2 | Chebyshev |
| `ML/Data_herm`, `ML/Data_herm_typ2` | 100×100 | 1 / 2 | Hermite (not used in final paper results) |
| `ML/data_unq_perm1..4`, `ML/data_typ2_perm1..4` | 100×100 | 1 / 2 | Legendre, lead-permuted (paper §4.2 sensitivity check) |

`ML/50_50_ml.ipynb` and several notebooks under `ML/data_stack*`, `ML/onebelowanother/`
hardcode an absolute path from the original author's machine
(`/home/karansingh/Documents/summer-term/ECG_ML/MAIN`) in a `sys.path.append(...)` call and are
**not runnable as-is** in this repository — this predates the reorganization and was not
changed. `ML/image_visulisation.ipynb` similarly hardcodes an old absolute data path.

## `save_data.ipynb`
Utility notebook that exports processed splits/intermediate datasets from `data_prep/` — kept
here since it reads `data_prep/...` by bare relative name and is now a direct sibling of it.

## Why `data_prep` and `ML` still appear (as symlinks) all over `run_experiments/`

Every experiment notebook references these two folders by bare name (`"data_prep"`,
`"ML/data_unq"`, etc.) and resets its own working directory back to wherever it lives via a
captured `default_dir`. Rather than edit that logic, each `run_experiments/<experiment>/`
folder gets a `data_prep` and `ML` symlink pointing back here — same data, zero duplication,
zero notebook edits.
