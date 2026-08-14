# Invertibility Analysis

This folder contains experiments and results for checking whether ECG signals encoded into polynomial-based 2D representations can be reconstructed back into the original 1D form.

## Directory Structure

```
invertibility/
├── Notebooks
│   ├── invertibility_typ1.ipynb           # Type 1 encoding/invertibility analysis
│   └── invertibility_typ2.ipynb           # Type 2 encoding/invertibility analysis
├── Visualizations
│   ├── histograms_TYP1.png                # Reconstruction metrics histogram (Type 1)
│   ├── histograms_TYP1_overlay.png        # Overlaid metric comparison (Type 1)
│   ├── histograms_TYP2.png                # Reconstruction metrics histogram (Type 2)
│   └── histograms_TYP2_overlay.png        # Overlaid metric comparison (Type 2)
├── Results - Type 1 Data
│   ├── result_of_TYP1_ST.csv              # Reconstruction metrics: ST samples
│   ├── result_of_TYP1_ST_all_samples.csv  # Reconstruction metrics: ST (all samples)
│   ├── result_of_TYP1_SB.csv              # Reconstruction metrics: SB samples
│   ├── result_of_TYP1_SB_all_samples.csv  # Reconstruction metrics: SB (all samples)
│   ├── result_of_TYP1_SR.csv              # Reconstruction metrics: SR samples
│   └── result_of_TYP1_SR_all_samples.csv  # Reconstruction metrics: SR (all samples)
├── Results - Type 2 Data
│   ├── result_of_typ2_ST.csv              # Reconstruction metrics: ST samples
│   ├── result_of_typ2_ST_all_samples.csv  # Reconstruction metrics: ST (all samples)
│   ├── result_of_typ2_SB.csv              # Reconstruction metrics: SB samples
│   ├── result_of_typ2_SB_all_samples.csv  # Reconstruction metrics: SB (all samples)
│   ├── result_of_typ2_SR.csv              # Reconstruction metrics: SR samples
│   └── result_of_typ2_SR_all_samples.csv  # Reconstruction metrics: SR (all samples)
├── results_combined.csv                   # Summary combining all Type 1 & Type 2 results
└── README.md
```

## Contents

- `invertibility_typ1.ipynb`: analysis for Type 1 ECG encoding/invertibility experiments.
- `invertibility_typ2.ipynb`: analysis for Type 2 ECG encoding/invertibility experiments.
- `histograms_TYP1.png` and `histograms_TYP2.png`: histogram summaries of the reconstruction metrics.
- `histograms_TYP1_overlay.png` and `histograms_TYP2_overlay.png`: overlaid visual comparisons for the same metrics.
- `result_of_TYP1_*.csv` and `result_of_typ2_*.csv`: reconstruction quality metrics for Type 1 and Type 2 datasets.
- `results_combined.csv`: combined summary of the invertibility experiments.

## Purpose

These notebooks and result files evaluate how well encoded ECG representations preserve the original signal information and whether reconstruction errors remain within acceptable limits.
