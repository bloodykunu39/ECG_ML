# Exploratory Notebooks (Archived)

These notebooks are early/scratch explorations, not part of the formal experiment
set reported in `draft.pdf` (Section 3 Results / Section 4.2). They were moved
here to declutter `MAIN/` without deleting anything.

- **`cg_200.ipynb`** — Exploratory test of coarse-graining ECG images down to a
  200x200 resolution (the paper only reports 100x100 and 50x50, see Sections
  3.1/3.2). Its data-loading cells already reference a path from an older
  repo layout (`../../KARAN_ECG/data_prep/...`) that does not exist in this
  repository, so it was not runnable as-is even before this move.
- **`superpostion_inverse.ipynb`** — Exploratory notebook on overlapping
  waveforms / signal reconstructibility, an early precursor to the more
  complete analysis now in `MAIN/invertibility/` (paper Section 3.9). It also
  references the old `../KARAN_ECG/data_prep/...` path and was not runnable
  as-is in this repository.

No content was changed — these are the original files, just relocated. If you
want to run them, you'll need to point their data-loading cells at
`MAIN/data_prep/` instead of the old `KARAN_ECG` path.
