# Experiment Index (mapped to draft.pdf)

This folder organizes the paper's experiments (Section 3 Results + Section 4.2)
into one subfolder per experiment. Each subfolder contains symlinks back to the
actual run notebook(s) in `MAIN/` and the corresponding `EXPERIMENT_*` results folder(s),
plus a short README describing what it is and which paper section it supports.

**Nothing was moved or modified** — the real notebooks, `.py` modules, and data folders
remain exactly where they were, since the notebooks use relative imports/paths assuming
`MAIN/` as the working directory. This index exists purely to make navigation easy.

| Paper § | Experiment | Notebook(s) | Results |
|---|---|---|---|
| 3.1.1 | [OPI (Legendre) benchmarking, CNN, 100x100, Data Type 1](./3.1.1_OPI_Legendre_Type1_CNN100/) | main_2d_leg_typ1.ipynb | EXPERIMENT_2d_leg_typ1_smallcnn_randomseed |
| 3.1.2 | [OPI (Legendre) benchmarking, CNN, reduced 50x50, Data Type 1](./3.1.2_OPI_Legendre_Type1_CNN50/) | main_2d_leg_on_50_typ1.ipynb | EXPERIMENT_2d_leg_50_typ1_smallcnn_randomseed |
| 3.2.1 | [OPI (Legendre) benchmarking, CNN, 100x100, Data Type 2](./3.2.1_OPI_Legendre_Type2_CNN100/) | main_2d_leg_typ2.ipynb | EXPERIMENT_2d_leg_typ2_smallcnn_randomseed |
| 3.2.2 | [OPI (Legendre) benchmarking, CNN, reduced 50x50, Data Type 2](./3.2.2_OPI_Legendre_Type2_CNN50/) | main_2d_leg_on_50_typ2.ipynb | EXPERIMENT_2d_leg_50_typ2_smallcnn_randomseed |
| 3.3.1 | [Baseline 1: 2D ECGResNet (ImageNet-pretrained), Data Type 1](./3.3.1_Baseline1_ResNet2D_Type1_Pretrained/) | main_2d_resnet_typ1.ipynb | EXPERIMENT_2d_resnet_typ1_ECGresnet_randomseed |
| 3.3.2 | [Baseline 1: 2D ECGResNet (randomly initialized), Data Type 1](./3.3.2_Baseline1_ResNet2D_Type1_NonPretrained/) | main_2d_resnet_typ1_pretrained_False.ipynb | EXPERIMENT_2d_resnet_typ1_ECGresnet_pretrain_FALSE_randomseed |
| 3.3.3 | [Baseline 1: 2D ECGResNet (ImageNet-pretrained), Data Type 2](./3.3.3_Baseline1_ResNet2D_Type2_Pretrained/) | main_2d_resnet_typ2.ipynb | EXPERIMENT_2d_resnet_ECGresnet_randomseed |
| 3.3.4 | [Baseline 1: 2D ECGResNet (randomly initialized), Data Type 2](./3.3.4_Baseline1_ResNet2D_Type2_NonPretrained/) | main_2d_resnet_typ2_pretrained_False.ipynb | EXPERIMENT_2d_resnet_typ2_ECGresnet_pretrain_FALSE_randomseed |
| 3.4.1 | [Baseline 2: Vanilla Transformer, Data Type 1](./3.4.1_Baseline2_VanillaTransformer_Type1/) | main_1_transformer_typ1.ipynb | EXPERIMENT_2d_transformer_typ1_VanillaTransformerECG_randomseed |
| 3.4.2 | [Baseline 2: Vanilla Transformer, Data Type 2](./3.4.2_Baseline2_VanillaTransformer_Type2/) | main_1_transformer_typ2.ipynb | EXPERIMENT_2d_transformer_typ2_VanillaTransformerECG_randomseed |
| 3.5.1 | [Baseline 3: Single-lead FFNN (Lead 1), Data Type 1](./3.5.1_Baseline3_SingleLeadFFNN_Type1/) | main_1_ffnn_singlelead_typ1.ipynb | EXPERIMENT_1d_firstlead_ffnn_typ1_randomseed |
| 3.5.2 | [Baseline 3: Single-lead FFNN (Lead 1), Data Type 2](./3.5.2_Baseline3_SingleLeadFFNN_Type2/) | main_1_ffnn_singlelead_typ2.ipynb | EXPERIMENT_1d_firstlead_ffnn_typ2_randomseed |
| 3.6.1 | [Baseline 4: 12-lead raw ECG, SelfEEG ResNet1D, Data Type 1](./3.6.1_Baseline4_12Lead1DResNet_Type1/) | main_1d_resnet_typ1_selfeeg.ipynb | EXPERIMENT_resnet_1d_selfeeg_typ1_randomseed |
| 3.6.2 | [Baseline 4: 12-lead raw ECG, SelfEEG ResNet1D, Data Type 2](./3.6.2_Baseline4_12Lead1DResNet_Type2/) | main_1d_resnet_typ2_selfeeg.ipynb | EXPERIMENT_resnet_1d_selfeeg_typ2_randomseed |
| 3.7.1 | [Baseline 5: Single-lead raw ECG, SelfEEG ResNet1D, Data Type 1](./3.7.1_Baseline5_SingleLead1DResNet_Type1/) | main_1d_resnet_typ1_selfeeg_singlelead1.ipynb | EXPERIMENT_resnet_1d_selfeeg_typ1_singlelead1_randomseed |
| 3.7.2 | [Baseline 5: Single-lead raw ECG, SelfEEG ResNet1D, Data Type 2](./3.7.2_Baseline5_SingleLead1DResNet_Type2/) | main_1d_resnet_typ2_selfeeg_singlelead1.ipynb | EXPERIMENT_resnet_1d_selfeeg_typ2_singlelead1_randomseed |
| 3.8.1 | [Alternative encoding: Chebyshev-CNN, Data Type 1](./3.8.1_AltEncoding_Chebyshev_Type1/) | main_2d_cheb_typ1.ipynb | EXPERIMENT_2d_cheb_typ1_smallcnn_randomseed |
| 3.8.2 | [Alternative encoding: Chebyshev-CNN, Data Type 2](./3.8.2_AltEncoding_Chebyshev_Type2/) | main_2d_cheb_typ2.ipynb | EXPERIMENT_2d_cheb_typ2_smallcnn_randomseed |
| 3.9 | [Invertibility of the OPI encoding](./3.9_Invertibility/) | ../invertibility/invertibility_typ1.ipynb<br>../invertibility/invertibility_typ2.ipynb | ../invertibility |
| 4.2 | [Discussion 4.2: Sensitivity to lead-to-polynomial assignment, Data Type 1](./4.2_LeadPermutation_Sensitivity_Type1/) | main_2d_leg_typ1_permutation.ipynb | EXPERIMENT_2d_leg_typ1_smallcnn_randomseed<br>EXPERIMENT_2d_leg_typ1_smallcnn_randomseed_permutation |
| 4.2 | [Discussion 4.2: Sensitivity to lead-to-polynomial assignment, Data Type 2](./4.2_LeadPermutation_Sensitivity_Type2/) | main_2d_leg_typ2_permutation.ipynb | EXPERIMENT_2d_leg_typ2_smallcnn_randomseed<br>EXPERIMENT_2d_leg_typ2_smallcnn_randomseed_permutation |

## Not covered above (methods / pipeline / exploratory, kept in place)

- **Section 2.1 Data Processing / Preparation:** `MAIN/data_prep/` (data_prep.ipynb, data_prep_for_only_one_disease.ipynb), `MAIN/zenodo_data/`
- **Section 2.1.3 Superposition, Coarse-Graining, Encoding:** `MAIN/encoding.py`, `MAIN/smoothening.py`, `MAIN/ML/` (smooth_using_normlised_legendre.ipynb, smooth_using _normlised _cehbyshev_and _hermite.ipynb, etc.)
- **Core modules (shared across all experiments above):** `MAIN/dataloader.py`, `dataseperation.py`, `encoding.py`, `model_cnn.py`, `model_nn.py`, `resnet1d.py`, `vanilla_transformer_ecg.py`, `plots.py`, `seed_utils.py`
- **Exploratory / utility notebooks (not tied to a specific paper section):** `MAIN/analysis.ipynb` (cross-experiment comparison dashboard, supports Discussion §4.1 comparisons), `MAIN/save_data.ipynb`, `MAIN/Extractor.ipynb`, `MAIN/12leadstack.ipynb`
- **Archived exploratory notebooks with stale/broken paths (moved, see `MAIN/exploratory_notebooks/README.md`):** `cg_200.ipynb`, `superpostion_inverse.ipynb`
