# ECG Classification using High Energy Physics-Inspired ML Techniques

This project focuses on classifying 12-lead Electrocardiogram (ECG) signals using machine learning techniques inspired by high-energy physics. We explore representation encoding, such as mapping 1D ECG signals into 2D spaces using Legendre, Chebyshev, and Hermite polynomials, and training deep learning models (CNNs, ResNets, Transformers) on these encoded representations.

---

## 📂 Project Structure & Directory Layout

### Core Directories

* **`ML/`**: Contains encoded ECG images and datasets, along with script files for downsampling and generation.
* **`data_prep/`**: Preprocessing scripts and filtered `.npy` files for raw ECG signals.
* **`final_output/`**: Stores finalized model checkpoints and outputs.
* **`invertibility/`**: Exploratory code testing the invertibility of encoded representations back to original 1D ECG signals.
* **`zenodo_data/`**: Raw and extracted data source folders.
* **`Example ECG/`**: Contains example raw 12-lead ECG reports (e.g., in `.mat` format).
* **`ARCHIVE/`**: Folder for deprecated, historical, or backup notebooks (such as `main_1_cnn*`, `main_cnn100_cheb*`, and `PatchTST`-based code).
* **`Other files/`**: Miscellaneous assets and scratch documents.

### Experiment Result Folders (`EXPERIMENT_*`)
These directories store result plots (accuracy curves, loss curves, confusion matrices, PR/ROC curves) and summary CSVs across multiple seeds for each notebook:
* `EXPERIMENT_1d_firstlead_ffnn_typ1_randomseed/`
* `EXPERIMENT_1d_firstlead_ffnn_typ2_randomseed/`
* `EXPERIMENT_2d_cheb_typ1_smallcnn_randomseed/`
* `EXPERIMENT_2d_cheb_typ2_smallcnn_randomseed/`
* `EXPERIMENT_2d_leg_50_typ1_smallcnn_randomseed/`
* `EXPERIMENT_2d_leg_50_typ2_smallcnn_randomseed/`
* `EXPERIMENT_2d_leg_typ1_smallcnn_randomseed/`
* `EXPERIMENT_2d_leg_typ2_smallcnn_randomseed/`
* `EXPERIMENT_2d_resnet_ECGresnet_randomseed/`
* `EXPERIMENT_2d_resnet_typ1_ECGresnet_randomseed/`
* `EXPERIMENT_2d_resnet_typ1_ECGresnet_pretrain_FALSE_randomseed/`
* `EXPERIMENT_2d_resnet_typ2_ECGresnet_pretrain_FALSE_randomseed/`
* `EXPERIMENT_2d_transformer_typ1_VanillaTransformerECG_randomseed/`
* `EXPERIMENT_2d_transformer_typ2_VanillaTransformerECG_randomseed/`
* `EXPERIMENT_resnet_1d_selfeeg_typ1_randomseed/`
* `EXPERIMENT_resnet_1d_selfeeg_typ2_randomseed/`
* `EXPERIMENT_resnet_1d_selfeeg_typ1_singlelead1_randomseed/`
* `EXPERIMENT_resnet_1d_selfeeg_typ2_singlelead1_randomseed/`

---

## 📑 Experiment & Utility Notebooks

### Model Training Notebooks
These notebooks contain the main pipelines for training classifiers on 1D/2D representations:

| Notebook | Model / Architecture | Representation / Encoding | Input Shape | Data Type | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`main_2d_cheb_typ1.ipynb`](./main_2d_cheb_typ1.ipynb) | SmallCNN | Chebyshev Polynomials | 100×100 | Type 1 | Training pipeline on Type 1 data |
| [`main_2d_cheb_typ2.ipynb`](./main_2d_cheb_typ2.ipynb) | SmallCNN | Chebyshev Polynomials | 100×100 | Type 2 | Training pipeline on Type 2 data |
| [`main_2d_leg_typ1.ipynb`](./main_2d_leg_typ1.ipynb) | SmallCNN | Legendre Polynomials | 100×100 | Type 1 | Coarse-grained (cg50) Legendre Type-1 |
| [`main_2d_leg_typ2.ipynb`](./main_2d_leg_typ2.ipynb) | SmallCNN | Legendre Polynomials | 100×100 | Type 2 | Coarse-grained (cg50) Legendre Type-2 |
| [`main_2d_leg_on_50_typ1.ipynb`](./main_2d_leg_on_50_typ1.ipynb) | SmallCNN | Legendre Polynomials | 50×50 | Type 1 | Reduced 50x50 resolution |
| [`main_2d_leg_on_50_typ2.ipynb`](./main_2d_leg_on_50_typ2.ipynb) | SmallCNN | Legendre Polynomials | 50×50 | Type 2 | Reduced 50x50 resolution |
| [`main_2d_resnet_typ1.ipynb`](./main_2d_resnet_typ1.ipynb) | ECGResNet | 2D Representation | 100×100 | Type 1 | Pretrained = True configuration |
| [`main_2d_resnet_typ2.ipynb`](./main_2d_resnet_typ2.ipynb) | ECGResNet | 2D Representation | 100×100 | Type 2 | Pretrained = True configuration |
| [`main_2d_resnet_typ1_pretrained_False.ipynb`](./main_2d_resnet_typ1_pretrained_False.ipynb) | ECGResNet | 2D Representation | 100×100 | Type 1 | Pretrained = False configuration |
| [`main_2d_resnet_typ2_pretrained_False.ipynb`](./main_2d_resnet_typ2_pretrained_False.ipynb) | ECGResNet | 2D Representation | 100×100 | Type 2 | Pretrained = False configuration |
| [`main_1d_resnet_typ1_selfeeg.ipynb`](./main_1d_resnet_typ1_selfeeg.ipynb) | ResNet1D | Raw 12-lead ECG | (12, 5000) | Type 1 | Using selfeeg model framework |
| [`main_1d_resnet_typ2_selfeeg.ipynb`](./main_1d_resnet_typ2_selfeeg.ipynb) | ResNet1D | Raw 12-lead ECG | (12, 5000) | Type 2 | Using selfeeg model framework |
| [`main_1d_resnet_typ1_selfeeg_singlelead1.ipynb`](./main_1d_resnet_typ1_selfeeg_singlelead1.ipynb) | ResNet1D | Raw Single-lead ECG | (1, 5000) | Type 1 | Using selfeeg, Lead 1 only |
| [`main_1d_resnet_typ2_selfeeg_singlelead1.ipynb`](./main_1d_resnet_typ2_selfeeg_singlelead1.ipynb) | ResNet1D | Raw Single-lead ECG | (1, 5000) | Type 2 | Using selfeeg, Lead 1 only |
| [`main_1_transformer_typ1.ipynb`](./main_1_transformer_typ1.ipynb) | VanillaTransformerECG | 2D Representation | 100×100 | Type 1 | Transformer-based 2D model |
| [`main_1_transformer_typ2.ipynb`](./main_1_transformer_typ2.ipynb) | VanillaTransformerECG | 2D Representation | 100×100 | Type 2 | Transformer-based 2D model |
| [`main_1_ffnn_singlelead_typ1.ipynb`](./main_1_ffnn_singlelead_typ1.ipynb) | FFNN | Raw Single-lead ECG | (5000, 1) | Type 1 | Single-lead Feedforward Neural Network |
| [`main_1_ffnn_singlelead_typ2.ipynb`](./main_1_ffnn_singlelead_typ2.ipynb) | FFNN | Raw Single-lead ECG | (5000, 1) | Type 2 | Single-lead Feedforward Neural Network |

### Analysis & Utility Notebooks
* [`analysis.ipynb`](./analysis.ipynb): Diagnostic dashboard for compiling experiment metrics, comparing accuracies, and plotting aggregate charts.
* [`Extractor.ipynb`](./Extractor.ipynb): Notebook for computing features and exploring data samples.
* [`12leadstack.ipynb`](./12leadstack.ipynb): Evaluates stacking architectures across multiple leads.
* [`cg_200.ipynb`](./cg_200.ipynb): Tests coarse graining to size 200.
* [`superpostion_inverse.ipynb`](./superpostion_inverse.ipynb): Exploration notebook for overlapping waveforms and signal reconstructibility.
* [`save_data.ipynb`](./save_data.ipynb): Exports processed splits and intermediate datasets.

---

## 🛠️ Python Scripts & Modules

* **[`dataloader.py`](./dataloader.py)**: Dataloading utilities for loading datasets in PyTorch.
* **[`dataseperation.py`](./dataseperation.py)**: Standardizes dataset splits (train, validation, test) and labels.
* **[`encoding.py`](./encoding.py)**: Polynomial signal-to-image encoding algorithms (e.g., Legendre, Chebyshev transformations).
* **[`encoding2.py`](./encoding2.py)**: Alternative and updated encoding implementations.
* **[`model_cnn.py`](./model_cnn.py)**: Contains PyTorch structures for the 2D CNNs (e.g., `SmallCNN`, `SmallCNN50`).
* **[`model_nn.py`](./model_nn.py)**: Standard feedforward neural networks (FFNNs) baseline.
* **[`vanilla_transformer_ecg.py`](./vanilla_transformer_ecg.py)**: Transformer models designed for 2D ECG signals.
* **[`plots.py`](./plots.py)**: Handles metrics plotting, ROC/PR curves, and confusion matrix exports.
* **[`smoothening.py`](./smoothening.py)**: Coarse-graining module (`cg50` refers to 50 coarsegrain size).
* **[`seed_utils.py`](./seed_utils.py)**: Ensures model training repeatability across different randomized seeds.

---

## 🚀 Getting Started & Execution

### Prerequisites
* Python 3.8+
* PyTorch 1.9+
* NumPy
* Pandas
* scikit-learn
* Matplotlib