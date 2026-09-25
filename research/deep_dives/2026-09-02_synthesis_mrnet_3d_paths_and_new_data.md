# Synthesis — MRNet, 3D paths, and how new data can move the final score

This note pulls together the research from 2026-09-02 into a single actionable picture.

## What the project already knows

1. **External-data seat axis is closed.** Concatenating external studies into the fine-tuning cache
   produces at best +0.006 and at six external seats −0.004. The corpus a member trained on is not
   what its seat is worth.

2. **External-data trunk axis is still open but bounded.** `OrthoFoundation-L` (1.25M knee images,
   multi-label supervision) was **declined**: fitted member 0.8452 < DINOv2-base control 0.8502.
   Knee-specific DINOv3 continuation can subtract signal. The same may not hold for 3D diffusion or
   for a carefully regularised supervised pretrain, but the prior is negative.

3. **The readout may matter more than the trunk.** DINOv2-L beats DINOv3-L and OrthoFoundation-L
   under the default max-pooled ridge probe, but Q-AL literature shows the same frozen trunk can
   move by ~0.11 AUC when the readout is made local. Bead `knee-02vn` tracks the re-probe.

4. **Pathology-specific SOTA models are not actionable.** No public foundation model covers a
   single RSNA pathology with weights. The only public multi-pathology knee models are MRNet,
   Rijeka, fastMRI+ and OrthoDiffusion.

## The three live bets

### Bet A: MRNet supervised pretrain → fine-tune on RSNA

- **Mechanism:** initialise the encoder from weights trained on MRNet's 1,370 knee exams (abnormal /
  ACL / meniscus), then fine-tune on RSNA.
- **Why it could help:** MRNet is domain-matched and provides dense knee-pathology supervision.
  Literature (Yosinski, Matsoukas, Azizi) says transfer gains live in lower/mid-level feature reuse,
  which would help all twelve RSNA labels, not just the three MRNet knows.
- **Why it might not:** MRNet is tiny compared with the pretraining that literature shows is needed
  for deep feature reuse. OrthoFoundation-L already failed at a much larger scale. The starting
  DINOv2/DINOv3 init is strong.
- **Honest ceiling:** per-label arithmetic on the three overlapping labels caps the mean-AUC gain at
  ~+0.0075; a broader representation gain is the optimistic case but likely small.
- **Experiment:** Claude has queued `experiments/arms/mrnet_acl_pretrain.sh` → `mrnet_warm.sh`.

### Bet B: OrthoDiffusion 3D backbone

- **Mechanism:** use the pretrained 3D diffusion U-Net from OrthoDiffusion (15,948 knee MRIs, MIT
  code, HuggingFace weights) as a feature extractor, add a 12-label head.
- **Why it is the best 3D bet:** first public knee-specific multi-label-diagnosis 3D foundation
  model; 5 of its 8 labels map directly to RSNA.
- **Risks:** diffusion inference is not a single encoder pass; input format is 16 central slices at
  256×256; Kaggle dataset licence is "unknown" (prefer HuggingFace); only 5/8 labels map.
- **Honest ceiling:** unknown; a successful smoke test would be a new member candidate and a
  benchmark for any 3D architecture trained in-house.
- **Status:** bead `knee-6hfi` tracks the smoke test.

### Bet C: Segmentation auxiliaries

- **Mechanism:** generate meniscus / cartilage / joint-region masks from foundation segmenters
  (SAM-Med3D, SegmentAnyBone, VISTA3D, MRI-CORE) and feed them as extra input channels to the 2D
  classifier.
- **Why it could help:** attacks the OA / meniscus columns indirectly without requiring clean
  external classification labels.
- **Risks:** no public knee-specific segmentation foundation model with proven RSNA transfer;
  adding channels changes the input tensor and requires a new training recipe.
- **Honest ceiling:** modest; likely a diversification add, not a main lever.

## What does NOT move the score

- More corpus-concatenation members (seat axis closed).
- Another 2D DINOv3 knee continuation (OrthoFoundation-L already lost).
- Chasing per-pathology SOTA classifiers (no public weights).
- Decipher-MR (no public weights).

## Recommended priority

1. **Run the readout re-probe (`knee-02vn`).** Zero card cost; may rewrite every trunk closure
   verdict.
2. **Smoke OrthoDiffusion (`knee-6hfi`).** Highest-ceiling untested public model.
3. **Run the MRNet pretrain pair (`knee-ombx`, `knee-1bjnh`).** Cheap and closes the supervised
   pretrain question.
4. **Scheme B merged multi-task partial-label trainer (`knee-n6tk`, `knee-mwuv`).** Uses existing
   decoded corpora with masked loss; cheaper than new architecture bets.
5. **Segmentation auxiliary only if 1-4 are positive and there is card time left.**

## Bottom line

The score is most likely to move from **(a) a better readout** or **(b) a 3D knee-specific
foundation model** than from more 2D trunk pretraining on small external corpora. MRNet pretrain is
worth the experiment because it is cheap and the literature says the mechanism is representation,
not head specialisation, but its ceiling is modest and the OrthoFoundation-L decline is the warning.
