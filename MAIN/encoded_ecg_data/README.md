
## Data and Encoding Processes

This section provides an overview of the data and the encoding processes used in this project.

### ECG Data Source

The raw, filtered ECG data is located in the `MAIN/data_prep` directory. This data serves as the input for our encoding processes.

### Data Encoding

The ECG data is transformed into image representations using a series of specialized encoding functions. These processes leverage orthogonal polynomials to generate distinct image types.

The primary encoding functions are:

- `smooth_using_normlised_legendre`: Encodes ECG signals using Legendre polynomials into 2D images
- `smooth_using_normlised_cehbyshev_and_hermite`: Encodes ECG signals using Chebyshev and Hermite polynomials into 2D images

### Directory Structure

```
encoded_ecg_data/
├── Legendre Encoding (100x100 resolution)
│   ├── data_leg_typ1/                  # Type 1: Single diagnosis Legendre images
│   ├── data_leg_typ2/                  # Type 2: Multi-diagnosis Legendre images
│   ├── data_leg_typ1_50/               # Type 1: Reduced resolution (50x50)
│   ├── data_leg_typ2_50/               # Type 2: Reduced resolution (50x50)
│   ├── data_leg_typ1_perm1/            # Type 1: Lead permutation 1
│   ├── data_leg_typ1_perm2/            # Type 1: Lead permutation 2
│   ├── data_leg_typ1_perm3/            # Type 1: Lead permutation 3
│   └── data_leg_typ1_perm4/            # Type 1: Lead permutation 4
├── Hermite Encoding (100x100 resolution)
│   ├── Data_herm/                      # Type 1: Single diagnosis Hermite images
│   └── Data_herm_typ2/                 # Type 2: Multi-diagnosis Hermite images
├── Chebyshev Encoding (100x100 resolution)
│   ├── Data_cheb/                      # Type 1: Single diagnosis Chebyshev images
│   └── Data_cheb_typ2/                 # Type 2: Multi-diagnosis Chebyshev images
├── smooth_using_normlised_legendre.ipynb            # Legendre encoding pipeline
├── smooth_using _normlised _cehbyshev_and _hermite.ipynb  # Chebyshev & Hermite encoding
├── encoded_ecg_image_visualization.ipynb.ipynb      # Visualization and analysis of encoded images
├── permutation.json                    # Channel permutation definitions
└── README.md
```

### Encoded Datasets

The encoded data is organized into specific directories based on the encoding method, image type, and resolution.

#### Legendre Encoding

| Directory | Image Resolution | Type | Description |
|---|---|---|---|
| `encoded_ecg_data/dat
a_leg_typ1` | 100x100 | 1 | Legendre encoded images of Type 1. |
| `encoded_ecg_data/dat
a_leg_typ2` | 100x100 | 2 | Legendre encoded images of Type 2. |
| `MAIN/encoded_ecg_data/dat
a_leg_typ1_50` | 50x50 | 1 | Legendre encoded images of Type 1, at a reduced resolution. |
| `MAIN/encoded_ecg_data/dat
a_leg_typ2_50` | 50x50 | 2 | Legendre encoded images of Type 2, at a reduced resolution. |

#### Legendre Encoding with Channel Permutations

Channel permutations applied to Type 1 Legendre datasets to explore robustness to different lead orderings:

| Directory | Type | Permutation | Description |
|---|---|---|---|
| `MAIN/encoded_ecg_data/dat
a_leg_typ1_perm1` | 1 | Permutation 1 | Legendre Type 1 with shuffled ECG lead order (see `permutation.json` for mapping). |
| `MAIN/encoded_ecg_data/dat
a_leg_typ1_perm2` | 1 | Permutation 2 | Legendre Type 1 with shuffled ECG lead order (see `permutation.json` for mapping). |
| `MAIN/encoded_ecg_data/dat
a_leg_typ1_perm3` | 1 | Permutation 3 | Legendre Type 1 with shuffled ECG lead order (see `permutation.json` for mapping). |
| `MAIN/encoded_ecg_data/dat
a_leg_typ1_perm4` | 1 | Permutation 4 | Legendre Type 1 with shuffled ECG lead order (see `permutation.json` for mapping). |

The channel permutations are defined in [`permutation.json`](./permutation.json).

#### Hermite Encoding

| Directory | Image Resolution | Type | Description |
|---|---|---|---|
| `MAIN/encoded_ecg_data/Data
_herm` | 100x100 | 1 | Hermite encoded images of Type 1. |
| `MAIN/encoded_ecg_data/Data
_herm_typ2` | 100x100 | 2 | Hermite encoded images of Type 2. |

#### Chebyshev Encoding

| Directory | Image Resolution | Type | Description |
|---|---|---|---|
| `MAIN/encoded_ecg_data/Data
_cheb` | 100x100 | 1 | Chebyshev encoded images of Type 1. |
| `MAIN/encoded_ecg_data/Data
_cheb_typ2` | 100x100 | 2 | Chebyshev encoded images of Type 2. |