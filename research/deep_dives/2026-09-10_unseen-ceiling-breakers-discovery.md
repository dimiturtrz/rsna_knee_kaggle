# 2026-09-10 — Unseen ceiling-breakers: a discovery sweep outside everything already priced

Owner-directed discovery search: approaches NOT in the project's priced set (trunk swaps,
PEFT, unfreeze ladders, resolution/extent, ensembles, distillation, pseudo-labels, label-noise
losses, ordinal/ASL losses, multi-plane input, ROI crops, harmonisation, TTA family, external
fine-tuning, contrastive DA, augmentation, mixup, slice-pooling operators, calibration/
uncertainty heads). Every entry: delta vs named baseline, corpus size, [CLF]/[SEG]/[ASSERTED].
Ranked by delta × corpus-size proximity to 4.4k studies. Source: research-lane search 2026-09-10;
this file is the permanent record. Not committed by this lane; provenance = letter D307.

## Ranked harvest

1. **Active label cleaning, prioritised relabel — Bernhardt et al., Nat Commun 2022**
   (PMC8897392) [CLF]. The canonical measurement of what re-labelling a fraction buys:
   prioritised re-annotation corrects labels up to **4× more efficiently than random**
   (NoisyCXR η=12.7%, N≈30k relabel simulations); a noisy validation set at 40% noise
   **reversed model ranking** (46.5% apparent vs 65.8% true) — the ruler-side argument.
   Mechanism is N-agnostic; same-decade evidence.

2. **RetClean — automatic label cleaning, Lin et al., npj Digit Med 2025** [CLF]. Six
   fully-automatic Cleanlab-style cycles on fundus/OCT: label accuracy +3.4–62.9%;
   RETFound retrained on cleaned data **+0.3–52.9% accuracy**, dataset-dependent
   (per-dataset N flagged unconfirmed). Same-scale evidence, wide range.

3. **Cross-corpus transfer, knee→knee — Bien et al. 2018 (MRNet)** [CLF]. Stanford
   sagittal-T2 model on external Rijeka (n=183): **0.824 zero-shot → 0.911 after training
   on the external data (+0.087)**. Same task family, N=1,130.

4. **Slice-attention + clinical-prior plane fusion — Kunitomi et al., GCCE 2025** [CLF].
   N=1,130 MRNet: slice-level attention pooling + plane fusion initialised from clinical
   priors (0.45/0.45/0.10): **ACL 0.910 vs 0.852 reimplemented MRNet baseline (+0.058)**.
   The unpriced novelty is prior-initialised learnable plane fusion; baseline sits below
   the original MRNet paper, so the delta is likely inflated.

5. **Frozen foundation-model features + linear probe on OAI knee MRI — Demir et al.,
   CVPRW 2026** [CLF]. Exactly MSK MRI, no fine-tuning, no distillation: SAM-Med3D
   features AUC 84.5 on KLG (aligned); uniGradICON 81.8 unaligned; progression detection
   75.6 (+13.6 over affine-only). Two transferable negatives: a 0.98-Dice task-specific
   U-Net produced the WEAKEST downstream features ("good segmentation ≠ good features"),
   and generic registration features beat the knee-specific specialist.

6. **Report-as-privileged-information distillation — Yue & Johansson** (arXiv 2510.24385,
   N=211–2,119) [CLF]. Contains the key NEGATIVE: report-aligned pretraining helps only
   when the label is well-represented in the text; when the label is fully derivable from
   the report, a much stronger multimodal teacher gives **no gain over self-distillation**.
   RSNA knee labels ≈ report content → predicted-negative for us; worth pricing once
   because the competition ships reports.

7. **Coarse-to-fine attention localisation on 9-label knee — Xie et al., eClinicalMedicine
   2025** [CLF]. N=8,891 train: removing AOLM drops per-label AUCs 0.730–0.966 →
   0.687–0.953 → learned attention localisation worth ~+0.01–0.04 per label. Adjacent to
   the excluded ROI-crop axis (learned, not a crop).

8. **k-space active acquisition policy — HieraSample** (arXiv 2607.19779) [CLF]. Mamba
   policy picks high-frequency k-space points; +20.4 AUC on ACL severity at 10× vs ASSDM
   Cartesian baseline, fastMRI+. Acquisition-side lever — closes an undersampling gap, does
   not lift a fully-sampled ceiling. Unpriced transferable idea: frequency-hierarchical
   training signals as regulariser.

9. **OrthoFoundation** (arXiv 2601.18250, Jan 2026 preprint) [CLF]. DINOv3-L SSL on 1.2M
   knee X-ray+MRI (OAI 200k MRI + fastMRI 300k + internal 400k): ACL 97.7% AUC at 50%
   labels; PCL 84.3% (+4 over next best). **Already known to this project and the
   licence/checkpoint chase was explicitly closed — listed here for completeness, not as
   a new find.**

## Measured negatives worth keeping

- k-space feature channels concatenated to an image CNN (Alzheimer's, N=8,131): no
  significant advantage; worse adjacent-grade separation at convergence.
- Multi-dataset MIL benchmark (3D neuroimage): frozen ViT-B/16 does better with plain
  Max/Mean pooling; learnable MIL heads only win with ConvNeXt — FocusMIL's claim that
  attention MIL violates the MIL assumption echoes from WSI.
- CurBench (NeurIPS 2024): curriculum methods frequently fail to beat standard training
  across benchmarks.
- CleanPatrick (Fitzpatrick17k audit): 22% label errors, 21% near-duplicates measured;
  automated label-error detection "remains an open challenge for fine-grained medical
  classification" — the automation half of the label-cleaning family is weak exactly at
  our regime.

## SILENT — searched, nothing exists

- **Joint multi-corpus knee training** (MRNet + KneeMRI/Rijeka + OAI + fastMRI-knee in one
  pool, partial label overlap, domain-tagged): silent in the PUBLISHED literature — but
  **this project built the cell and measured it null** (`merged3_corpus`, −0.0024, ACL
  supervision column −0.0013; see `2026-09-10_unseen-candidates-mapped-onto-our-measurements.md`).
  Read that mapping before treating any silent cell here as "unbuilt".
- Learned slice-sampling/selection policies (RL/sequential) for study-level MSK MRI: nothing.
- Curriculum-SSL or anatomy-aware SSL pretexts designed for knee/MSK with public downstream
  numbers: only a numberless ISMRM abstract.
- k-space augmentation with a disease-classification endpoint on knee: nothing.
- Coreset/pruning moving a classification ceiling at ~4k studies: nothing medical with
  numbers (DUCS is nearest, unverified).
- Structured-report supervision (non-CLIP, non-distillation) on MSK MRI: nothing.
- Ordinal-hierarchical multitask beyond OAI KL-grading: nothing; Xie 2025 uses independent
  per-label models.

## Flagged unverifiable

DUCS (no numbers extracted) · ISMRM #4014 OAI-DINO (registrant-gated) · KMAE (k-space
classification numbers not extracted) · RetClean per-dataset Ns · OrthoFoundation (preprint,
internal downstream set) · Kunitomi baseline-inflation caveat · an unidentified OpenReview
preprint claiming +~0.05 AUC after teacher retraining.
