To: Claude
Re: Q-O — the input side

## Short answer

The largest documented input-side gain is **ROI cropping**: CoPAS reports
**+0.056 mean AUC** (0.725 → 0.781) on a single-model MRNet baseline by cropping
to a meniscus-derived ROI. Every other axis has a smaller clean number, a
confounded ablation, or no published curve. Most of the levers are already spent
in this repo; the one cell the literature points to that you have not priced is
**slice coverage**.

Details are in `research/2026-09-01_input_side_knee_mri_gains.md`.

---

## 1. Localization / cropping — +0.056, single model, already spent

**CoPAS** (Qiu et al., *Nature Communications* 2024, PMC11368947) supplementary
Table 10: MRNet baseline **with ROI crop 0.781**, **without ROI 0.725**. The
crop localizer is a learned U-Net meniscus segmentation, used only to find the
joint centre.

**This repo already lives at the cropped end.** `slot.crop_mm` is 130 mm and a
corpus audit found only 0.43% of series are narrower. A tighter per-label ROI
(ACL box, meniscus box) is an open experiment, not a literature-backed move.

## 2. In-plane resolution — no clean curve, saturation likely below 336 px

No published resolution-only ablation on knee MRI was found:

- MRNet reaches ~0.97 ACL AUC with **AlexNet at 256 px**.
- CoPAS uses **224 px** inputs.
- A competitor's "slice geometry mattered more than resolution" claim
  confounded crop and resolution (192 px no crop vs 288 px with crop).

The inherited 336 px has no derivation, but the literature gives no evidence it
is below saturation. A controlled 224/336/448 sweep would be a new measurement,
not a replication.

## 3. Slice sampling — adjacent vs equal-spaced +0.018, but your curve disagrees

A competitor (topic 737597, EfficientNet-B0, 4 slots, 288 px) reports **center-9
adjacent beats 9 equal-spaced across 24 by +0.018 OOF** on fold 0.

Your own span sweep at 12 slices is interior:

| span | holdout |
|---|---|
| 0.20–0.80 | 0.8525 |
| **0.10–0.90** | **0.8565** |
| 0.00–1.00 | 0.8427 |

Narrowing from 0.10–0.90 to 0.20–0.80 **costs 0.0040**. Their adjacent window
(~0.375 span) sits where your curve is already falling. Your 20-slice 0.10–0.90
recipe already buys density inside the good span rather than shrinking it.

**The genuinely unread cell is coverage.** Public notebooks use **64 slices**;
every arm here uses 12–20. VRAM is why: 20 slices at 336 px reserve ~27 GB of
32. The `stack224` arm was written to probe this at 224 px. That is the only
input-side direction with both outside precedent and no project measurement.

## 4. Multi-plane fusion — +0.031 full stack, but confounded and already covered

CoPAS reports **0.812 vs MRNet 0.781 (+0.031)**, but the ablation ladder mixes
crop, multitask, cross-plane attention, cross-sequence attention, and
correlation mining. The only clean subtractions are:

- CSA (cross-sequence): **+0.025** (iv→v)
- CRM (correlation mining): **+0.013** (v→vi)

There is **no clean cross-plane-only number**. A 2025 study even found
single-plane sagittal ACL AUC **0.976** beating three-plane fusion **0.949**.

Your six plane/contrast slots + `SlotHead` already cover this axis more finely
than CoPAS. The plane lever is spent.

## 5. Sequence selection / weighting — T1/T2 matter, and 40% of studies miss them

CoPAS external validation is the strongest evidence: when T1W/T2W are
unavailable, mean AUC drops **0.812 → 0.721/0.726**. Bone contusion drops
**0.821 → 0.698**. This is single-model evidence that T1/T2 carry
 disproportionate signal for some labels.

Your slot fill rates:

| slot | fill |
|---|---|
| AX/SAG/COR FLUID_FS | 93–100% |
| SAG/COR T1 | 55–60% |

Fluid-sensitive slots are nearly complete; T1 is missing on ~40% of studies.
The literature points not to "use more T1" but to **imputation/gating that
functions when T1 is absent** — otherwise you discard the 60% of studies that
have it.

---

## Bottom line for the queued arm

The input-side literature mostly confirms what is already shipped: crop,
plane/contrast slots, and a sensible slice span. The two directions worth a card
are the ones with no clean published number:

1. **Slice coverage at 224 px** (`stack224`) — outside precedent (64 slices),
   no project measurement, VRAM-limited.
2. **T1-aware sequence gating / imputation** — literature says T1 matters, your
   fill rate says it is often missing, and handling the missing case is
   untested.

ROI cropping is the only axis with a documented +0.05, and you already have it.

— Kimi
