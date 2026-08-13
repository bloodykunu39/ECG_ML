# Run Experiments — Index (mapped to draft.pdf)

Each subfolder below is a **real, self-contained working directory** for one experiment
reported in the paper (`../../draft.pdf`, Results §3 and Discussion §4.2): it physically
contains the run notebook and its own results (`EXPERIMENT_*` folder with per-seed logs,
plots, and summary CSVs).

The `.py` module files and the `data_prep`/`ML` folders you see inside each subfolder are
**symlinks** back to `../../src/` and `../../data_preparation/` — the single real copies.
This is needed because these notebooks capture their own directory as `default_dir` at
startup and `os.chdir()` back to it between cells, then load data / write results with
bare relative names (`data_prep`, `ML/...`, `EXPERIMENT_...`) — so each notebook needs its
dependencies to actually be present alongside it. The symlinks make that true without
duplicating any data or editing a single line of notebook code.

| Paper § | Experiment | Notebook | Owns |
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
| 4.2 | [Discussion 4.2: Sensitivity to lead-to-polynomial assignment, Data Type 1](./4.2_LeadPermutation_Sensitivity_Type1/) | main_2d_leg_typ1_permutation.ipynb | EXPERIMENT_2d_leg_typ1_smallcnn_randomseed_permutation |
| 4.2 | [Discussion 4.2: Sensitivity to lead-to-polynomial assignment, Data Type 2](./4.2_LeadPermutation_Sensitivity_Type2/) | main_2d_leg_typ2_permutation.ipynb | EXPERIMENT_2d_leg_typ2_smallcnn_randomseed_permutation |

See also (kept in their own top-level folders, already self-contained):

- **§3.9 Invertibility** → `../invertibility/` (`invertibility_typ1.ipynb`, `invertibility_typ2.ipynb`)
