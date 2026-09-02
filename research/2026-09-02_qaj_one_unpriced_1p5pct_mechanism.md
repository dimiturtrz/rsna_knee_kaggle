# Q-AJ — one mechanism that could carry +1.5%

Question (Claude, 2026-09-02): name ONE published mechanism not on the closed list that could
deliver ≥+1.5% mean AUC on our 12-label knee-MRI task with ~4,400 labeled studies.

## Honest short answer

The published frontier for our exact task shape does **not** contain a clean, already-measured
+1.5% AUC mechanism that is not on your closed list. The strongest remaining candidate is
**domain-specific self-supervised continuation on the external knee-MRI corpora** (fastMRI +
MRNet + Rijeka), but the knee-MRI literature only shows modest or same-label gains, not a
>1.5% multi-label AUC delta.

## What was already priced or demoted

Your closed list covers member quality, ensemble diversity, slice aggregators, overlap TTA,
trunk family, pseudo-labeling, masked partial labels, report-label repair, folds/seeds, and
calibrated targets/curriculum. The literature audit below was checked against that list.

## Candidate 1 — domain-specific SSL continuation (the best remaining bet)

Existing notes: `research/2026-09-01_ssl_continuation_from_strong_init.md`,
`research/2026-09-01_external_knee_mri_corpus_transfer.md`, and the new
`research/2026-09-02_external_mri_data_merging_schemes.md`.

Published numbers closest to our shape:

| Study | Source | Target | Delta | Metric |
|---|---|---|---|---|
| Azizi et al. 2021, *Big Self-Supervised Models Advance Medical Image Classification* | ImageNet SSL + unlabeled MIMIC-CXR | CheXpert multi-label chest X-ray | **+1.1% mean AUC** | mean AUC |
| Ilse et al. 2025, *Data Scaling Laws for Radiology Foundation Models* | 100k–3.5M in-domain chest X-rays | in-domain chest tasks | surpasses open-weights; rib-fracture AUPRC +6% | AUPRC / classification |
| Li et al. 2025, *MedDINOv3* | DINOv3 + 3.87M CT slices | AMOS22 segmentation | **+1.07–1.87% DSC** | DSC |
| Manna et al. 2021, *SKID* | MRNet (1,370 clips) | KneeMRI ligament labels | **+0.020 AUC** | binary AUC |
| Qiu et al. 2025, *PRISM* | 336k multi-sequence MRI incl. knee | MRNet same labels | **+2.27% accuracy** | accuracy, AUC not reported |

Why it is the best bet:
- It directly consumes the external data we already have access to.
- The ceiling is **unmeasured** for knee→knee partial-label multi-label AUC; no paper prices that exact cell.
- It is objective-matched: continue a DINO-family trunk with a DINO-family SSL loss on in-domain knee MRI.

Why it is not a guaranteed +1.5%:
- Zhang et al. 2024 (`arXiv:2403.16499`) found fastMRI in-domain SSL **matched but did not beat**
  ImageNet supervised initialization on MRNet labels.
- SKID's cross-target knee gain is only +0.020 AUC.
- PRISM reports accuracy, not AUC, and on same labels.

## Candidate 2 — Sharpness-Aware Minimization (SAM)

Papers found:
- Mariam et al. 2024, retinal vessel segmentation: AUC **0.8908 → 0.9839 (+10.4%)**.
- Sukegawa et al. 2022, oral exfoliative cytology: large accuracy/AUC gains with SAM + LR scheduler.

Why it is NOT the answer:
- Both are small binary/segmentation tasks with simple CNNs trained from scratch or weak init.
- No published SAM result on a pretrained DINO/ViT medical classifier was found.
- ImageNet ViT-B SAM gain is ~+1.0% top-1; transfer to our task is speculative and likely smaller.

## Candidate 3 — slice coverage beyond 20 slices

`research/2026-09-01_input_side_knee_mri_gains.md` flags this as the only unread input-side cell,
but no published number prices it. It is an experiment, not a literature-backed mechanism.

## What to tell Claude

The honest answer to Q-AJ is: **no published mechanism with a clean >1.5% AUC for our exact task
shape is left unpriced**. If one experiment must be armed, the least-confounded bet with a
non-zero ceiling is **domain-specific SSL continuation on fastMRI + MRNet + Rijeka**, started
from our best DINO-family trunk, using masked loss for partial labels and RSNA target-domain
oversampling. It should be treated as an exploration, not a +1.5% promise.
