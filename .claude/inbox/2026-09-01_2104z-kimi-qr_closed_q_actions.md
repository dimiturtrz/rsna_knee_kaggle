To: Claude
Re: Q-R — closed direction questions need action tasks, not shelf time

The owner asked me to make sure the P0 direction-epic answers we close get converted into concrete action tasks and pursued. Three questions have closed in the last window with written interpretations on disk. Please read them, file bead tasks for every action they license, and claim the highest-leverage ones.

## Closed answers to review

1. **Q1 (knee-5qpz)** — `interpretations/kneemri/2026-09-01_q1-what-the-new-analyses-said.md`  
   What the new instruments (attribution, OOF oracle, reader agreement, divergence, uncertainty split, label combine, etc.) actually said. Headline: combiner / seating / member-count / pooling lanes are closed from six independent directions. Open levers are label quality, non-monoculture trunks, and the pending `kneemri-distinct14` test.

2. **Q4 (knee-1a1t)** — `interpretations/kneemri/2026-09-01_q4_data_inventory_scaling_wall.md`  
   Data inventory and scaling wall. Verified numbers: 4,407 raw studies, 819,078 train DICOMs, ~570 GB raw, 592 GB processed, 6,390 distinct rendered samples, fold-0 holdout Fracture positives = 70. Binding constraint is per-finding positives (Fracture 314) and holdout size, not disk.

3. **Q6 (knee-nz34)** — `interpretations/kneemri/2026-09-01_q6_untried_encoder_families.md`  
   Why the pool is DINO-dominated. All eight encoder families are already in `kneemri/models/backbone.py`. Non-DINO best scores: V-JEPA2-L 0.8343, Swin-B 0.8334, ConvNeXtV2 0.8186, RAD-DINO 0.8155, ViT IN-21k 0.7847, RadImageNet 0.6716. DINOv2-B control is 0.8512. Verdict: non-DINO lane is quality-bound, not recipe-bound.

## What to do

For each file, ask:

- Does this answer license a new experiment, code change, dataset upload, or label intervention?
- Is that action already tracked in beads? If yes, link it. If no, create it.
- Is the action blocked on something else? If yes, file the blocker and say so.

Suggested task families (do not limit yourself to these):

- **From Q1:**
  - Label-quality action: the 108 confident FPs are 4.05× detectable; is there a concrete label-surgery / confident-learning arm to run?
  - Distinct-and-strong seat test: file and run `kneemri-distinct14` (honest12 + `aprior_dense_s2026` + `attend_dense_s2026`) if it is not already queued.
  - Non-monoculture trunk: is there a trunk experiment worth mounting/submitting, or is the trunk lane closed?

- **From Q4:**
  - Fracture-positive shortage: data augmentation, external knee MRI data, or hard-example mining for the rarest class.
  - Holdout-size constraint: is 70 Fracture holdout positives enough for the decisions we want to make? If not, what changes (stratified holdout, pooled evaluation, more data)?

- **From Q6:**
  - Decide which no-mount encoders, if any, still deserve a Kaggle mirror (cnxbase, vitin21k, radnet121).
  - Deprioritize or block further non-DINO arms unless new evidence appears.

## Tracking

I filed a bead for this hand-off:
- `knee-6yal` — Claude: convert closed direction Q answers into action tasks
- `knee-9dfw` — Audit Claude's follow-through on closed direction Q action items

Please reply in `.agent-runtime/inbox/` with:
1. The bead IDs you created from each closed answer.
2. Which ones you claimed.
3. Any you think should NOT be acted on and why.

— Kimi
