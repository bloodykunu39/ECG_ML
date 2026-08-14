# ECG Machine Learning - Main Project Directory

This directory contains the core pipeline and experiments for ECG signal classification using polynomial-based 2D image encodings and various deep learning models.

## Overview

The project workflow consists of:
1. **Data Preparation**: Filter and organize raw ECG data from PhysioNet
2. **Encoding**: Transform 1D ECG signals into 2D polynomial-based image representations
3. **Model Training**: Train various CNN, Transformer, and ResNet architectures
4. **Analysis**: Evaluate model performance and signal invertibility

---

## Directory Structure

```
MAIN/
├──  Core Data Processing
│   ├── data_prep/
│   │   ├── data/                    # Raw PhysioNet ECG database
│   │   ├── data_prep.ipynb          # Filter and organize raw data
│   │   ├── data_prep_for_only_one_disease.ipynb  # Type 1 dataset creation
│   │   ├── disease_*.npy            # Type 2 processed arrays
│   │   ├── unq_disease_*.npy        # Type 1 processed arrays
│   │   └── README.md
│   ├── encoding.py                  # Core encoding utility functions
│   └── smoothening.py               # Signal smoothing utilities
│
├──  Encoding & Transformation
│   ├── encoded_ecg_data/
│   │   ├── Legendre encodings       # 100x100 & 50x50 2D images
│   │   ├── Hermite encodings        # 100x100 2D images
│   │   ├── Chebyshev encodings      # 100x100 2D images
│   │   ├── smooth_using_normlised_legendre.ipynb
│   │   ├── smooth_using_normlised_cehbyshev_and_hermite.ipynb
│   │   ├── encoded_ecg_image_visualization.ipynb.ipynb
│   │   ├── permutation.json         # Channel permutation mappings
│   │   └── README.md
│   ├── superpostion_inverse.ipynb   # Signal superposition analysis
│   └── invertibility/               # Reconstruction quality analysis
│       ├── invertibility_typ1.ipynb
│       ├── invertibility_typ2.ipynb
│       ├── histograms_*.png
│       ├── result_of_*.csv          # Reconstruction metrics
│       └── README.md
│
├──  Model Architectures & Utilities
│   ├── model_cnn.py                 # Small CNN architecture
│   ├── model_nn.py                  # Feedforward neural network
│   ├── resnet1d.py                  # 1D ResNet implementation
│   ├── vanilla_transformer_ecg.py   # Transformer architecture for ECG
│   ├── dataloader.py                # Custom PyTorch data loaders
│   ├── dataseperation.py            # Train/val/test split utilities
│   ├── seed_utils.py                # Random seed management
│   ├── plots.py                     # Visualization utilities
│   └── analysis.ipynb               # Cross-model analysis & results
│
├──  Training Notebooks - 1D Approaches (Single Lead)
│   ├── main_1_ffnn_singlelead_typ1.ipynb     # Feedforward NN on first lead (Type 1)
│   ├── main_1_ffnn_singlelead_typ2.ipynb     # Feedforward NN on first lead (Type 2)
│   ├── main_1_transformer_typ1.ipynb         # Transformer on first lead (Type 1)
│   ├── main_1_transformer_typ2.ipynb         # Transformer on first lead (Type 2)
│   ├── main_1d_resnet_typ1_selfeeg.ipynb     # 1D ResNet all leads (Type 1)
│   ├── main_1d_resnet_typ2_selfeeg.ipynb     # 1D ResNet all leads (Type 2)
│   ├── main_1d_resnet_typ1_selfeeg_singlelead1.ipynb    # 1D ResNet first lead (Type 1)
│   └── main_1d_resnet_typ2_selfeeg_singlelead1.ipynb    # 1D ResNet first lead (Type 2)
│
├──  Training Notebooks - 2D Approaches (Encoded Images)
│   ├── Legendre Encoding (100x100)
│   │   ├── main_2d_leg_typ1.ipynb            # Type 1 training
│   │   ├── main_2d_leg_typ2.ipynb            # Type 2 training
│   │   ├── main_2d_leg_typ1_permutation.ipynb   # Channel permutation robustness (Type 1)
│   │   └── main_2d_leg_typ2_permutation.ipynb   # Channel permutation robustness (Type 2)
│   │
│   ├── Legendre Encoding (50x50 Resolution)
│   │   ├── main_2d_leg_on_50_typ1.ipynb      # Type 1 training (reduced res)
│   │   └── main_2d_leg_on_50_typ2.ipynb      # Type 2 training (reduced res)
│   │
│   ├── Chebyshev Encoding (100x100)
│   │   ├── main_2d_cheb_typ1.ipynb           # Type 1 training
│   │   └── main_2d_cheb_typ2.ipynb           # Type 2 training
│   │
│   ├── ResNet-50 (2D Images)
│   │   ├── main_2d_resnet_typ1.ipynb         # Legendre Type 1 (pretrained)
│   │   ├── main_2d_resnet_typ1_pretrained_False.ipynb   # Legendre Type 1 (no pretrain)
│   │   ├── main_2d_resnet_typ2.ipynb         # Legendre Type 2 (pretrained)
│   │   └── main_2d_resnet_typ2_pretrained_False.ipynb   # Legendre Type 2 (no pretrain)
│   │
│   └── Transformer (2D Images)
│       ├── main_2d_transformer_typ1.ipynb    # Legendre Type 1
│       └── main_2d_transformer_typ2.ipynb    # Legendre Type 2
│
├──  Experiment Results
│   ├── EXPERIMENT_*_randomseed/     # 20+ experiment folders containing:
│   │   ├── model_weights/           # Trained model checkpoints
│   │   ├── results/                 # Predictions, metrics, logs
│   │   ├── experiment_config.json   # Hyperparameter records
│   │   └── seed_record.txt          # Random seed for reproducibility
│   │
│   └── Organized by: [encoding method]_[type]_[architecture]_[variants]
│
├──  Supporting Materials
│   ├── Example ECG/
│   │   ├── data_1.mat               # Sample ECG record 1
│   │   └── data_2.mat               # Sample ECG record 2
│   │
│   ├── Other files/
│   │   ├── images.ipynb             # Quick figure generation
│   │   ├── image_paper.ipynb        # Publication-quality figures
│   │   ├── 12_lead_ecg.png          # ECG format illustration
│   │   ├── 12_lead_ecg_encoding.png # Encoding process diagram
│   │   ├── feedforward_nn.png       # NN architecture diagram
│   │   ├── small_cnn_*.png          # CNN architecture variations
│   │   ├── 2dimage.png              # 2D image example
│   │   ├── image_1D_multiple_signals.png
│   │   └── README.md
│   │
│   ├── requirements.txt             # Python package dependencies
│   └── analysis.ipynb               # Cross-experiment comparison
```

---

## Key Components

### Data Preparation Pipeline
- **Input**: Raw 12-lead ECG records from PhysioNet
- **Output**: Numpy arrays organized by disease (ST, SB, SR) and type (Type 1/2)
- **See**: `data_prep/README.md`

### Encoding Methods
Three orthogonal polynomial-based 2D image encodings:
- **Legendre**: Standard encoding with multiple resolutions (100x100, 50x50)
- **Hermite**: Alternative polynomial basis
- **Chebyshev**: Alternative polynomial basis
- **See**: `encoded_ecg_data/README.md`

### Model Architectures

**1D Approaches** (operate directly on ECG signals):
- Feedforward Neural Networks (1D signals)
- 1D ResNet (SelfEEG-based)
- Transformer (on 1D signals)

**2D Approaches** (operate on encoded images):
- Small CNN (custom architecture)
- ResNet-50 (pretrained & from scratch)
- Transformer (vision-based)

### Dataset Types
- **Type 1**: Single diagnosis per patient (pure/controlled data)
- **Type 2**: Multi-diagnosis per patient (real-world complexity)

### Experiment Organization
Each EXPERIMENT folder contains:
- Model checkpoints and weights
- Training/validation/test results
- Performance metrics (accuracy, precision, recall, F1, confusion matrices)
- Configuration and seed logs for reproducibility

---

## File Organization by Purpose

**Model Training**: `main_*.ipynb` notebooks
**Core Utilities**: `*.py` modules (encoding, models, data handling)
**Data**: `data_prep/` and `encoded_ecg_data/`
**Analysis**: `analysis.ipynb`, `invertibility/`
**Figures**: `Other files/`
**Examples**: `Example ECG/`

---

## Quick Start

1. **Prepare data**: Run `data_prep/data_prep.ipynb`
2. **Encode signals**: Run `encoded_ecg_data/smooth_using_normlised_legendre.ipynb`
3. **Train models**: Select a training notebook (e.g., `main_2d_leg_typ1.ipynb`)
4. **Analyze results**: Check experiment folders or run `analysis.ipynb`

For detailed information on each stage, see the README files in the respective subdirectories.

---

## Reproducibility

All experiments use fixed random seeds stored in:
- `seed_utils.py`: Seed management utilities
- `EXPERIMENT_*/seed_record.txt`: Specific seeds used per experiment

Reproduce results by running the same training notebook with the same seed.
