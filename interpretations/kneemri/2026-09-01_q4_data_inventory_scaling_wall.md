# Q4 — Data inventory and scaling wall

**Question:** How large is the knee-MRI corpus, what is actually available to train on, and where does the binding constraint sit?

## Raw competition corpus

| Item | Count / Size |
|------|--------------|
| `train.csv` studies | 4,407 |
| `test.csv` studies | 3 |
| `train_series/` DICOM files | 819,078 |
| `test_series/` DICOM files | 557 |
| Total raw files (CSVs + DICOMs) | ~819,640 |
| `train_series/` size | 569.2 GB |
| `test_series/` size | 0.6 GB |
| Median files / train study | 162 |
| Median size / train study | 79.4 MB |

The train/test split is fixed by the competition. Test is essentially a token (3 studies); every model is scored on the public/private leaderboard, not on a project holdout.

## Processed artefacts

| Item | Count / Size |
|------|--------------|
| `processed/rsna_knee_abnormality/` files | 89,832 |
| `processed/rsna_knee_abnormality/` size | 592 GB |
| Label table used at train time (`labels/report_labels_v2.csv`) | 4,406 studies |

The processed tree is larger than raw because it holds the same studies rendered at multiple resolutions, slice bands, and decoder settings, plus checkpoints, holdout records, and experiment caches.

## Distinct rendered samples vs. re-renders

The 592 GB processed directory does **not** contain 89k independent patients. The distinct rendered samples are:

| Source | Studies |
|--------|---------|
| RSNA competition (`report_labels_v2.csv`) | 4,406 |
| Rijeka ACL (`labels/rijeka_acl.csv`) | 736 |
| MRNet ACL (`labels/mrnet_acl.csv`) | 1,248 |
| **Distinct rendered samples** | **6,390** |

The union label table `merged3_labels.csv` contains 6,390 rows, confirming the additive count. The 58-study `gold58` annotation set is a separate radiologist-adjudicated ruler; it is not training data.

## Labels actually used for training

`report_labels_v2.csv` carries the 12 findings with confidence/verdict columns. Positives by verdict `YES` (the hard target most members train on):

| Finding | Positives |
|---------|----------:|
| Fracture | 314 |
| Synovitis | 580 |
| MCL | 767 |
| Lateral OA | 1,087 |
| Baker's | 1,114 |
| Lateral Meniscus | 1,121 |
| ACL | 1,226 |
| Contusion | 1,297 |
| Medial OA | 1,619 |
| PF OA | 1,991 |
| Medial Meniscus | 2,347 |
| Effusion | 2,666 |

The rarest finding, Fracture, drives the statistical-power floor.

## Fold-0 split

`core.data.corpus.folds.ReportFolds` assigns by report-text hash so duplicate templates never straddle train/holdout. Fold 0 (default seed 2026):

| Split | Studies |
|-------|--------:|
| Training | 3,497 |
| Holdout | 909 |

Holdout positives per finding:

| Finding | Holdout positives |
|---------|------------------:|
| Fracture | **70** |
| Synovitis | 120 |
| MCL | 171 |
| Lateral Meniscus | 228 |
| Lateral OA | 236 |
| ACL | 249 |
| Baker's | 218 |
| Contusion | 273 |
| Medial OA | 334 |
| PF OA | 409 |
| Medial Meniscus | 484 |
| Effusion | 560 |

The Fracture holdout count is exactly 70 because that was the pre-registered constraint; the hash-based fold happened to satisfy it.

## Scaling wall

The competition corpus has **4,406 training studies**. With Rijeka and MRNet merged, that rises to **6,390 distinct samples**, but only the ACL label is enriched by the external corpora; the other eleven findings still see only 4,406 studies.

A useful literature anchor is Thian et al., "Effect of training data volume on the performance of deep learning algorithms in medical imaging" ([PMC9485337](https://pmc.ncbi.nlm.nih.gov/articles/PMC9485337/)). On chest-radiograph pneumothorax (a binary radiology task), they found AUC rose quickly from 2k to 20k training samples, then improved only gradually up to 291k. EfficientNet-B2 reached AUC 0.92 at 20k and 0.98 at 291k; ResNet-50 reached 0.86 at 20k and 0.95 at 291k.

Knee MRI is not a direct transplant: each "sample" is a multi-slice, multi-series volume with 12 concurrent labels, and the rarest positive class has only 314 cases. Still, the paper gives a rough order-of-magnitude boundary. Even the full merged corpus (6.4k studies) sits below the 20k-image knee of the Thian curve, and for Fracture the effective sample size is the number of positives (314) and holdout positives (70). That is the binding constraint: not disk space, not DICOM count, but **per-finding positive counts and the size of the holdout set they can support**.

## Verdict

- Disk inventory: ~570 GB raw, ~592 GB processed, 89k+ processed files.
- Distinct samples: ~6,400, of which only 4,406 carry all twelve findings.
- The scaling wall is **label-limited**, especially for Fracture (314 positives, 70 holdout positives), not storage-limited.
