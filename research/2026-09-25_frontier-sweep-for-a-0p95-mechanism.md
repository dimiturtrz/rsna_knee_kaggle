# Frontier sweep for a 0.95+ mechanism (2026-09-25)

Question: is there a published, licence-clean mechanism NOT already on our axes that could plausibly move
a solo by ≥ +0.01, given our board 0.946 (rank 97/4318, read 2026-09-25T05:52Z) against a leader at 0.959?

Scope and limits: Kaggle discussion / write-up pages render by JavaScript and returned only titles, so no
thread from 2026-09-20 on and no lumbar-2024 1st-place write-up was read. Kaggle-side claims come from
GitHub repos and search snippets and are marked as such.

## This competition — the public ceiling sits BELOW us

| Source | Public LB | Mechanism | Already ours? |
|---|---|---|---|
| "Speedy Raptors" notebook (search snippet only) | 0.943 | timm encoders, per-sequence ("protocol") embeddings, BiGRU over slices, per-finding transformer queries with masked attention pooling; per [Hisernberg/knee](https://github.com/Hisernberg/knee) solo arms 0.932 / 0.927, forks flat (0.942) | **Yes** — `SlotHead.slot_emb` + per-label `query` (`kneemri/models/slot_head.py:62-63`) |
| [TranBaDat2607](https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection) | 0.939 | CoAtNet ensemble; LLM re-extracted report labels 0.813 vs rule-based 0.809 on gold-58 (their own SE ≈ 0.02) | Noise; no lever |
| [ShivenKhurana1](https://github.com/ShivenKhurana1/rsna-knee-abnormality) | 0.935 | DINOv2/CNN + BiGRU; +0.0010 from a 4th model family, per-target blend −0.0008, label consensus −0.0054 | Confirms blend tricks are flat (matches our closed seat axes) |
| [JunhaoLiXD](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection) | 0.664 | fluid-sensitive-first series priority; soft labels with positive > unmentioned > negative ordering | Low value |

Nothing public on this competition beats 0.946. The leader's method is not public.

## Transfers from earlier RSNA challenges (untested on knee)

| Mechanism | Source | Reported gain | Status here |
|---|---|---|---|
| Segmentation auxiliary loss (UNet decoder + dice on the last two blocks) | [RSNA 2023 abdominal, 1st](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution) | +0.01 to +0.03 on **log loss**, not AUC | Only documented ≥ +0.01 lever. Needs knee masks, which exist only in foreign sets (OAI-ZIB, SKM-TEA) → **instrument-only** under the data ruling; cannot ship |
| Two-stage locate → crop at higher resolution | [RSNA 2024 lumbar, 2nd](https://github.com/brendanartley/RSNA-2024-Competition) | keypoints within 5 px 99.1% of the time | **Tested, lost.** `ROI_SLOTS` APPENDS the sagittal crops (`core/data/corpus/slots.py:389-405`): roi 0.8984 / roinative 0.9018 vs no-crop 0.9051 ledger75 (knee-vgz5b) |
| Confident-learning label denoising, 0.5 real + 0.5 pseudo loss | same | no number quoted | **Done** — cleanlab ranking `experiments/README.md:7745`; `relabel7_v1` board 0.943 |
| Manifold mixup, slice-order flip, 9-rotation TTA | same | no number quoted | mixup **CLOSED** (`docs/AXES.md:138`); order flip ≈ slice_shift (−0.0016) |
| Per-plane separate pipelines | same | +0.01–0.02 CV solo, flat in their ensemble | Our trunk already reads all six slots jointly |

Papers: AnyMC3D (arXiv 2512.12887): 2D foundation model + ~1M-param task plugins beats 3D models. This is essentially our current design. No peer-reviewed weakly-supervised knee-report result was found to replicate.

## What survives

1. **Registered slice selection (knee-bps3x, in-house).** The cache draws slices as a FRACTION of the
   acquired stack (`core/data/corpus/slot_reader.py:328`), so slice k names a different plane per study.
   A physics-based fix places the sagittal band on the knee's L-R centre from cross-plane
   `ImagePositionPatient` geometry, which is licence-clean. It is discounted because the shipping line's depth
   attention is deliberately position-free and slice_shift read flat. Estimate: P(≥ +0.01) ≈ 0.03–0.05, cost ≈ 4–6 h.
2. **Segmentation auxiliary loss as an INSTRUMENT.** It answers whether dense spatial supervision moves this trunk at
   all. A positive result would license a hunt for a licence-clean mask source; it can never be shipped itself.

Verdict: the sweep found no licence-clean, orthogonal, published ≥ +0.01 mechanism. The gap to 0.959 is
not public knowledge.
