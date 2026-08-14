# Data Preparation

This directory contains the scripts and information related to the preparation and processing of the raw ECG data used in this project. The primary goal of this stage is to filter and categorize the raw data before it is used for encoding and subsequent machine learning tasks.

---

### Data Source

The raw 12-lead ECG data was obtained from the [PhysioNet database](https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip).

### Directory Structure

```
MAIN/
├── data_prep/
│   ├── data/
│   │   ├── a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/
│   │   │   ├── ConditionNames_SNOMED-CT.csv          # Disease code mappings
│   │   │   ├── LICENSE.txt                           # Database license
│   │   │   ├── RECORDS                               # List of all patient records
│   │   │   ├── SHA256SUMS.txt                        # File integrity checksums
│   │   │   └── WFDBRecords/                          # Raw ECG data (46 patient batches)
│   │   │       └── 01/ - 46/                         # Numbered folders with .dat/.hea files
│   │   └── a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
│   ├── data_prep.ipynb                               # Raw data filtering & preprocessing
│   ├── data_prep_for_only_one_disease.ipynb          # Type 1 dataset creation
│   ├── disease_*.npy                                 # Type 2 processed data (multi-diagnosis)
│   ├── unq_disease_*.npy                             # Type 1 processed data (single diagnosis)
│   └── README.md
├── Example ECG/
│   ├── data_1.mat                                    # Sample ECG record 1 (.mat format)
│   └── data_2.mat                                    # Sample ECG record 2 (.mat format)
└── ...
```

### Data Preparation Scripts

The data preparation process is handled by the following Jupyter notebooks:

-   `data_prep.ipynb`: Filters raw PhysioNet data and organizes ECG records by disease category. Generates Type 2 datasets allowing multiple diagnoses per patient.
-   `data_prep_for_only_one_disease.ipynb`: Creates Type 1 datasets by selecting only patient records with a single disease code, ensuring pure single-diagnosis data for controlled model training.

### Dataset Types

The raw data is processed into two distinct dataset types for different experimental purposes:

-   **Type 1:** ECG records containing only a single disease code per patient. This dataset is specifically created for models trained on pure, single-diagnosis data.
-   **Type 2:** ECG records containing one primary disease code but potentially including additional comorbidities. This dataset is used to test model robustness on more complex, real-world data.

Filenames that begin with `unq` (e.g., `unq_disease_SR.npy`) correspond to **Type 1** data, while all other filenames (e.g., `disease_SR.npy`) correspond to **Type 2**.

### Corrupted Files

The following files were identified as corrupted during the data preparation process and were excluded from all datasets:
-   `23 236 JS23074`
-   `01 019 JS01052`

### Disease Categories and Sample Sizes

The project focuses on three specific disease categories. The sample sizes for both Type 1 and Type 2 datasets for these diseases are detailed in the table below.

| Disease | Code | Type 1 Sample Size | Type 2 Sample Size |
| :--- | :--- | :---: | :---: |
| ST | 427084000 | 3000 | 5000 |
| SB | 426177001 | 5000 | 5000 |
| SR | 426783006 | 5000 | 5000 |