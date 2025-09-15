# ECG Classification using High Energy Physics-Inspired ML Techniques

This project aims to classify 12-lead Electrocardiogram (ECG) images using machine learning techniques inspired by high energy physics.
Our approach leverages advanced algorithms and methodologies commonly used in particle physics to enhance the accuracy and efficiency of ECG interpretation.

## Project Overview

Electrocardiograms are crucial diagnostic tools in cardiology. 
However, their interpretation can be complex and time-consuming.
This project applies cutting-edge machine learning techniques,
drawing inspiration from high energy physics,to automate and 
improve the classification of 12-lead ECG images.

### Key Features

- Implementation of physics-inspired ML models for ECG classification
- Utilization of techniques like NN and CNN.
- Preprocessing pipeline for 12-lead ECG image data
- Performance comparison with traditional deep learning approaches

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
## Project Structure

| Folder/File | Description |
|-------------|-------------|
| `/ml/` | Contains image encoding scripts and encoded image data |
| `/data_prep/` | Data preparation scripts and filtered NPY files |
| `/results/` | Pickle files containing model results and analysis |

### Data Preparation

The data preparation phase includes:
- Data filtering into two categories
- NPY file generation
- Raw data processing scripts

### Model Results

The results directory contains pickle files with:
- Model performance metrics
- Classification results
- Evaluation statistics

## Experiment Notebooks

| Notebook                                | Model / Setup             | Data Encoding        | Input Shape | Data Type | Notes |
|-----------------------------------------|---------------------------|----------------------|-------------|-----------|-------|
| `main_1_cnn100_unique.ipynb`            | SmallCNN (100×100)        | Legendre             | 100×100     | Type 1    | Trained on Type-1 data |
| `main_1_cnn100.ipynb`                   | SmallCNN (100×100)        | Legendre             | 100×100     | Type 2    | Trained on Type-2 data |
| `main_1_cnn50_typ2.ipynb`               | SmallCNN50                | Legendre             | 50×50       | Type 2    | Explicit Type-2 data |
| `main_1_cnn50.ipynb`                    | SmallCNN50                | Legendre             | 50×50       | Type 2    | Trained on Type-2 data |
| `main_1_nn_cg16_leg_singlelead_typ2.ipynb` | Feedforward NN          | Raw ECG (Lead 1)     | (5000,1)    | Type 2    | Single-lead, Type-2 data |
| `main_1_nn_cg16_leg_singlelead.ipynb`   | Feedforward NN            | Raw ECG (Lead 1)     | (5000,1)    | Type 1    | Single-lead, Type-1 data |
| `main_1_nn_cg16_leg_unique.ipynb`       | Feedforward NN (flattened)| Legendre             | 100×100     | Type 1    | Trained on Type-1 data |
| `main_1_nn_cg16_leg.ipynb`              | Feedforward NN (flattened)| Legendre             | 100×100     | Type 2    | Trained on Type-2 data |
| `main_cnn100_cheb_typ2.ipynb`           | CNN (100×100)             | Chebyshev            | 100×100     | Type 2    | Trained on Type-2 data |
| `main_cnn100_cheb.ipynb`                | CNN (100×100)             | Chebyshev            | 100×100     | Type 1    | Trained on Type-1 data |
| `main_cnn100_herm_typ2.ipynb`           | CNN (100×100)             | Hermite              | 100×100     | Type 2    | Trained on Type-2 data |
| `main_cnn100_herm.ipynb`                | CNN (100×100)             | Hermite              | 100×100     | Type 1    | Trained on Type-1 data |
| `superpostion_inverse.ipynb`            | Utility                   | Various encodings    | -           | -         | Verifies reversibility of encoded images |


## Key Scripts

- **dataloader.py**  
  Custom PyTorch dataloader for handling ECG datasets.

- **dataseperation.py**  
  Scripts for filtering and separating the raw ECG data into different categories.
  
- **encoding.py**  
  Functions for encoding ECG signals into images using **Legendre polynomials**.

- **model_cnn.py**  
  Definitions of CNN architectures such as **SmallCNN**, **SmallCNN50**, and variants.

- **model_nn.py**  
  Definitions of **feedforward neural networks (FFNNs)** for ECG experiments.

- **plots.py**  
  Utility functions for plotting model evaluation results 
  
- **smoothening.py**  
    has coarsegrain function for downsampling the ECG Encoded images.


## Main Folders

- **ML/**  
  Contains encoded ECG images (from data originally in `.txt` format).  
  Also includes smoothing and encoding code used to generate these datasets.

- **data_prep/**  
  Data preparation scripts and filtered `.npy` files.  
  Includes documentation and notebooks for preprocessing raw ECG data.

- **ECG_ML/**  
  Subproject or additional code/data related to ECG machine learning.

- **Example ECG/**  
  Contains example raw ECG reports in `.mat` format.

- **main_1_cnn50/**  
  Experiment results and saved models for `main_1_cnn50.ipynb`.

- **main_1_cnn50_typ2/**  
  Experiment results and saved models for `main_1_cnn50_typ2.ipynb`.

- **main_1_nn_cg16_leg/**  
  Results and models from `main_1_nn_cg16_leg.ipynb`.

- **main_1_nn_cg16_leg_singlelead/**  
  FFNN experiment results and models for single-lead  leg data (`main_1_nn_cg16_leg_singlelead.ipynb`).

- **main_1_nn_cg16_leg_singlelead_typ2/**  
  Results and models for single-lead  leg Type-2 data.


note : cg16 means nothing at all it should be cg50 mean 50 coarsegrain
