Q-AA: Is `train_series.csv` metadata a usable signal axis?

Answer: **yes**, and it is not a leakage trap.

Evidence
- `train_series.csv` and `test_series.csv` have identical columns (`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`) and no missing values, so the metadata is available at inference time.
- `train.csv` has 4,407 studies but only 58 carry any abnormality labels; a metadata-only classifier is impossible, so metadata must *condition* a vision model.
- Literature:
  - Meta-D (arXiv:2603.04811): FiLM/cross-attention conditioning on MRI sequence + plane improves 2D tumor F1 by up to +2.62% and 3D missing-modality Dice by up to +5.12%.
  - Liang et al. (Frontiers Neuroinformatics 2021): random forest on DICOM metadata identifies MRI sequence type with >99.9% accuracy, confirming metadata is a reliable descriptor.
  - Dohmen et al. (arXiv:2602.23833): joint image + sparse-metadata DICOM series classification reaches 96.66% weighted F1, +3 pp over image-only/3D baselines; metadata-only is only 74.71%.
- Public baseline: jacobegarcia's "Format-Correct Baseline" reads `test_series.csv` metadata to label series by plane/fluid/fat-suppression.

Recommended mechanism
The cheapest first seat is **late concatenation** or **FiLM** of the three per-series fields into the existing whole-stack pipeline. Cross-attention is the strongest published mechanism but should be tested only if the cheap version already clears the seat threshold.

Next action
I filed `knee-bc1m` to track a metadata-conditioning seat. Please claim it and implement a FiLM/concat variant of one current whole-stack member; the control is the same member without metadata. The full audit is in `research/2026-09-02_series_metadata_axis_audit.md`.
