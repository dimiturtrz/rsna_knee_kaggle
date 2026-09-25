# Q7 (knee-brqy): external data usable for self-supervised pretraining

2026-09-01, Kimi. Read-only audit. The corpus survey in `research/deep_dives/2026-09-01_knee-mri-corpora-survey.md` is the main source; this note adds the tooling and cost side.

## What external knee-MRI data exists

| corpus | licence for Kaggle use | usable exams/scans | value for SSL | blockers |
|---|---|---|---|---|
| **KMAR-50K** | CC BY 4.0 — yes | 1,444 paired scans / 62k slices | encoder transfer only (no labels) | none; direct download |
| **fastMRI knee + fastMRI+** | images: NYU DUA (research only, no redistribution); labels: MIT | 10,012 DICOM exams | largest clean-ish pool; also study-level/bbox labels for 6 finding neighborhoods | each user needs NYU approval; data cannot ship in a kernel |
| **MRNet** | Stanford research-use agreement | 1,250 exams decoded on disk | labelled ACL/meniscus/MTC, but small | research-use only; already partially ingested |
| **KneeMRI / Rijeka** | unclear via Kaggle mirror | 736 exams on disk | ACL expert reads | mirror licence never recorded; verify before redistribution |
| **SKM-TEA** | Stanford DUA, non-commercial | 155 patients, ~25k slices | rich meniscus/ACL/cartilage bbox labels | DUA text gated; AIMI registration needed |
| **OAI / MOST** | DUC/OHDTA, no redistribution, 12-month deletion | 4,796 / 3,026+ subjects | longitudinal OA labels | institutional paperwork; trained-weights status ambiguous |
| **The competition training set itself** | competition rules | 4,407 studies | already the corpus `pretrain.py` reads; report labels exist but SSL can ignore them | none |

The largest **unlabelled** corpus that is genuinely free and shippable is **KMAR-50K**. The largest overall is **fastMRI knee**, but it is DUA-bound and cannot be mounted in a Kaggle kernel.

## What the repo already has for SSL

- `kneemri/training/pretrain.py` — masked spectral MAE over the cached slices; outputs a state dict consumable by `--encoder-weights`.
- `kneemri/training/spectral_mae.py` — spectral reconstruction term aimed at focal detail.
- `kneemri/evaluation/encoder_drift.py` — measures how far adapted blocks move from the stock reference.

The pretrainer draws **individual slices** rather than studies, so the same decode cache is roughly two orders of magnitude more SSL steps than supervised steps.

## Cost of the smallest useful SSL run

Using the current training cache (~3,498 studies, ~3 slots, ~12–20 slices each):

- slice population: ~3,498 × 3 × 12 ≈ **126k slices** at the old default, or ~210k at 20 slices.
- default `slices_per_step = 96` → ~1,300–2,200 steps per epoch.
- a supervised whole-stack epoch on the dense recipe is ~400 s; SSL has no slot head but a small decoder, so budget **similar or slightly faster** per step.
- **10 epochs ≈ 3–4 h of card** for the smallest run that sees the corpus once.

That is the price to prove the wiring. A run that could move the frontier would likely need **20–50 epochs plus external data**, so a realistic wall-clock estimate is **10–30 h** of card for one pretraining checkpoint, before any supervised arm is run on top of it.

## Go / no-go

- **GO — KMAR-50K as encoder-transfer corpus**: CC BY, no registration, ~62k slices, new population (Chinese, mixed 1.5T/3T). The project already has evidence that encoder transfer from an unlabelled corpus helps (`Lateral Meniscus +0.0196` from a corpus that never labels it). This is the cheapest external-data bet.
- **GO — re-run `pretrain.py` on the competition data alone**: the cache is already paid for; 3–4 h tests whether spectral MAE changes the frozen-block representation enough to matter. This should run before any external decode, because it tells us whether the SSL objective itself works on knee MRI.
- **CONDITIONAL — fastMRI**: apply for the DUA if the KMAR-50K + competition run shows signal, but do not plan a Kaggle kernel around it.
- **NO-GO — OAI/MOST for SSL**: the DUC overhead (PI, IRB, 12-month deletion) is only worth it if the smaller free corpora already prove the SSL path.

## Verdict

There is more usable data, but the **free and shippable** part is modest: KMAR-50K adds ~1,444 scans, and fastMRI adds 10k exams under a no-redistribution DUA. The practical next step is a **3–4 h spectral-MAE run on the existing competition cache** to validate the SSL objective, followed by ingesting KMAR-50K if that run moves `encoder_drift`.
