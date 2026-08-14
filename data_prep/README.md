# Data Preparation

This directory contains the scripts and information related to the preparation and processing of the raw ECG data used in this project. The primary goal of this stage is to filter and categorize the raw data before it is used for encoding and subsequent machine learning tasks.

---

### Data Source

The raw 12-lead ECG data was obtained from the [PhysioNet database](https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip).

### Directory Structure

```
data_prep/
├── data/
│   ├── a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/
│   │   ├── ConditionNames_SNOMED-CT.csv          # Disease code mappings
│   │   ├── LICENSE.txt                           # Database license
│   │   ├── RECORDS                               # List of all patient records
│   │   ├── SHA256SUMS.txt                        # File integrity checksums
│   │   └── WFDBRecords/                          # Raw ECG data (46 patient batches)
│   │       └── 01/ - 46/                         # Numbered folders with .dat/.hea files
│   └── a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
├── data_prep.ipynb                               # Raw data filtering & preprocessing
├── data_prep_for_only_one_disease.ipynb          # Type 1 dataset creation
├── disease_*.npy                                 # Type 2 processed data (multi-diagnosis)
├── unq_disease_*.npy                             # Type 1 processed data (single diagnosis)
└── README.md
```

Sample records for quick testing without the full dataset are in [`../example_data/`](../example_data).

### Data Preparation Scripts

The data preparation process is handled by the following Jupyter notebooks:

-   `data_prep.ipynb`: Filters raw PhysioNet data and organizes ECG records by disease category. Generates Type 2 datasets allowing multiple diagnoses per patient.
-   `data_prep_for_only_one_disease.ipynb`: Creates Type 1 datasets by selecting only patient records with a single disease code, ensuring pure single-diagnosis data for controlled model training.

### Dataset Types

The raw data is processed into two distinct dataset types for different experimental purposes:

-   **Type 1:** ECG records containing only a single disease code per patient. This dataset is specifically created for models trained on pure, single-diagnosis data.
-   **Type 2:** ECG records containing one primary disease code but potentially including additional comorbidities. This dataset is used to test model robustness on more complex, real-world data.

Filenames that begin with `unq` (e.g., `unq_disease_SR.npy`) correspond to **Type 1** data, while all other filenames (e.g., `disease_SR.npy`) correspond to **Type 2**.

### Corrupted / Excluded Samples

Two independent corruption checks are applied, at two different stages of the pipeline. Both matter for reconciling the sample counts below with the numbers reported in the paper.

**1. Corrupted source records** — excluded during `data_prep.ipynb`, before the `.npy` arrays below are ever written:
-   `23 236 JS23074`
-   `01 019 JS01052`

**2. NaN/Inf-corrupted samples** — a second, separate check run inside each `main_*.ipynb` training notebook, *after* loading `disease_*.npy` / `unq_disease_*.npy`. A sample is flagged if any value in its 12×5000 signal is NaN or Inf:

```python
def check_data(data):
    corrupted_indices = []
    for i in range(data.shape[0]):
        if np.isnan(data[i]).any() or np.isinf(data[i]).any():
            corrupted_indices.append(i)
    return corrupted_indices
```

The 1D notebooks (`main_1_ffnn_singlelead_*`, `main_1d_resnet_*_selfeeg*`, `main_2d_resnet_*`) call `check_data()` live on each run. The 2D encoded-image notebooks (`main_2d_leg_*`, `main_2d_cheb_*`) instead hardcode its output as a fixed index list (identical across every 2D notebook for a given Type) rather than recomputing it — if the upstream data ever changes, these lists would need regenerating via `check_data()`.

Per-class corrupted indices (0-indexed into `data_ST_list` / `data_SB_list` / `data_SR_list` as loaded, i.e. before concatenation):

| | Type 1 | Type 2 |
| :--- | :--- | :--- |
| **ST** | `[199, 306, 449, 633, 920, 2132, 2141, 2186, 2256]` | `[2341, 3205, 3257, 3746, 3810, 3837, 3928, 4232, 4302, 4317, 4516, 4533, 4578]` |
| **SB** | `[1418, 1852, 1898, 2756, 2881, 3007, 3101, 4091, 4192, 4293, 4376, 4594, 4601, 4720]` | `[3941, 4202, 4212, 4271]` |
| **SR** | `[149, 192, 213, 340, 385, 441, 2341, 2451, 2482, 2534, 4903, 4944]` | `[1854, 2166, 2256, 2833, 2888, 3212, 3375, 3749, 3817, 3820, 4083, 4236, 4253, 4425, 4875, 4896]` |

Critically, **these samples are still physically present in the `.npy` files** — they're skipped at training time in every notebook, not removed from disk. This is exactly what reconciles the "raw" sample counts below with the sample sizes reported in the paper's Table 1:

| Disease | Type 1: raw − corrupted = used | Type 2: raw − corrupted = used |
| :--- | :---: | :---: |
| ST | 3000 − 9 = **2991** | 5000 − 13 = **4987** |
| SB | 5000 − 14 = **4986** | 5000 − 4 = **4996** |
| SR | 5000 − 12 = **4988** | 5000 − 16 = **4984** |

### Disease Categories and Sample Sizes

The project focuses on three specific disease categories. The raw (pre-corruption-exclusion) sample sizes on disk for both Type 1 and Type 2 datasets are detailed in the table below — see [above](#corrupted--excluded-samples) for the post-exclusion counts actually used in experiments, which match the paper's Table 1.

| Disease | Code | Type 1 Sample Size (raw) | Type 2 Sample Size (raw) |
| :--- | :--- | :---: | :---: |
| ST | 427084000 | 3000 | 5000 |
| SB | 426177001 | 5000 | 5000 |
| SR | 426783006 | 5000 | 5000 |