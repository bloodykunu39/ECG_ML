
## Data and Encoding Processes

This section provides an overview of the data and the encoding processes used in this project.

### ECG Data Source

The raw, filtered ECG data is located in the `MAIN/data_prep` directory. This data serves as the input for our encoding processes.

### Data Encoding

The ECG data is transformed into image representations using a series of specialized encoding functions. These processes leverage orthogonal polynomials to generate distinct image types.

The primary encoding functions are:

- `smooth_using_normlised_legendre`
- `smooth_using_normlised_cehbyshev_and_hermite`

### Encoded Datasets

The encoded data is organized into specific directories based on the encoding method, image type, and resolution.

#### Legendre Encoding

| Directory | Image Resolution | Type | Description |
|---|---|---|---|
| `ML/data_unq` | 100x100 | 1 | Legendre encoded images of Type 1. |
| `ML/data_4` | 100x100 | 2 | Legendre encoded images of Type 2. |
| `MAIN/ML/data_unq_50` | 50x50 | 1 | Legendre encoded images of Type 1, at a reduced resolution. |
| `MAIN/ML/data_50` | 50x50 | 2 | Legendre encoded images of Type 2, at a reduced resolution. |

#### Hermite Encoding

| Directory | Image Resolution | Type | Description |
|---|---|---|---|
| `MAIN/ML/Data_herm` | 100x100 | 1 | Hermite encoded images of Type 1. |
| `MAIN/ML/Data_herm_typ2` | 100x100 | 2 | Hermite encoded images of Type 2. |

#### Chebyshev Encoding

| Directory | Image Resolution | Type | Description |
|---|---|---|---|
| `MAIN/ML/Data_cheb` | 100x100 | 1 | Chebyshev encoded images of Type 1. |
| `MAIN/ML/Data_cheb_typ2` | 100x100 | 2 | Chebyshev encoded images of Type 2. |