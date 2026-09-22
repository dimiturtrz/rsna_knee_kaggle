# kneemri — PLAN (source of truth)

This is the canonical plan. Read it at session start before proposing work. Substance lives here and is
cited by bead id; the ORDER and STATE of the work live in beads (`bd ready`, `bd list`), which is the one
place they can be true. Measured results live in `experiments/README.md` and `interpretations/`; the
build log is git history. **`docs/AXES.md` indexes every lever already tried** — verdict and pointer, one
row each — and is the first thing to read before proposing an axis.

## Goal

Twelve findings from one knee MRI study, for RSNA's
[knee abnormality detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition (deadline **2026-10-22**, 1012 teams at entry).

Portfolio-wise the deliverable is not a leaderboard row: it is a *reproduced frontier plus at least one
real idea*, with the honest account of which part is which. A grafted score is a floor, never a result.

## Headline metrics

The competition score is the **unweighted mean of twelve per-label ROC AUCs**. Three consequences, each
of which removes a design choice rather than adding one:

- **Only order matters.** AUC is invariant under any strictly increasing map, so calibration and
  thresholds are worth nothing, and models combine by averaging **ranks**, not probabilities (`knee-e8a`).
- **Every label costs the same.** A label at chance forfeits `(M−0.5)/12` — at `M=0.85`, 0.029, larger
  than most modelling gains. Rare findings therefore deserve *more* attention, not less (`knee-foh`).
- **Prevalence drift is survivable, thresholds are not.** Prevalence is not guaranteed to match across
  train / public / final; AUC is in expectation invariant to it.

| metric | baseline | target | current |
|---|---|---|---|
| mean of 12 AUCs (public LB) | 0.5 (constant) · 0.819 median · 0.888 graft | 0.942 (leader) | **0.919** (all-ours, two whole-stack members; rank 84 of 1758) |
| per-member holdout AUC (proxy ruler) | 0.8081 (small@8) | — | **0.8523** (`res392_ws_s2026`, our best single member) |
| fused holdout AUC | — | — | **0.8612** (`kneemri-chosen`, four members by joint holdout → 0.918 LB) |
| worst single-label AUC | 0.5 | — | **0.7981** Lateral Meniscus (±0.0159, 1114 positives — on the PUBLISHED members' corpus-wide out-of-fold predictions, the only per-label ruler that spans all 4,407 studies) |
| submission path works end to end | — | proven | proven (both scored) |

**The pool, not the configuration, carries the 0.919**: five all-ours members scored 0.911 and two scored
0.919, both clearing the 24-member mixed fusion (0.898) by more than the 1.5% bar, while restoring a
twenty-fifth published member changed nothing. Grafted members are now a COST (`knee-k5kk`).

Track the **worst label**, not only the mean — that is where the 1/12 is being forfeited.

**Two things about the second row, both established 2026-08-11 and both load-bearing:**

1. **It is measured with the wrong ruler.** `train.csv` carries the twelve radiologist annotations for
   exactly 58 studies (1.32% of cells, all-or-nothing), and RSNA states the evaluation set is
   radiologist-annotated. So every local number is agreement with LLM-read *report* labels, and it
   cannot be fixed locally — the 58 are trained on at weight 3.0, and the dozen in fold 0's holdout sit
   in the ±0.17 Hanley-McNeil regime. **The local number decides WHICH recipe to submit; only the
   leaderboard decides whether it is better.** The two ledger rows carrying both numbers suggest the
   proxy COMPRESSES by ~2x (local +0.0107 → LB +0.020), on two confounded points.
2. **A result needs ~1.5% absolute.** Almost nothing measured here has ever cleared that as a single
   change — not base (+0.0085), not 8→24 epochs (+0.0068), not the ranking term (+0.0067/+0.0036/+0.0028).
   They are *directional*. What clears the bar is the CUMULATIVE 0.8081 → **0.8523** = **+0.0442** on the
   holdout, and on the BOARD the one change that cleared it alone: dropping the published members for an
   all-ours pool, 0.898 → 0.919 (`knee-k5kk`).

   **Which floor applies depends on the ruler.** One fold of 881 studies has a Hanley–McNeil floor of
   0.0053; the published members' corpus-wide out-of-fold partition (4,407 studies) sits near 0.0024;
   and a PAIRED comparison of two recipes on the same studies is about 2x tighter than the unpaired
   bound, not the 11x an earlier estimate claimed (2026-08-15). Quote the floor belonging to the ruler
   the number was measured on.

## Approach

Public evidence → our-data statistics → method, in that order. The public evidence is read for
**mechanism**, and every borrowed constant is re-derived on our own corpus before it is trusted
(`knee-25n`); a constant that survives our measurement is a finding, one that does not is a bug we
would otherwise have inherited.

### What the task actually is

**Measured 2026-08-10: 4407 training studies, all 4407 carry a `Report`, and exactly 58 carry all
twelve annotations.** 1.3%. This is a weak-supervision problem first and an imaging problem second —
the labels for 98.7% of the corpus have to be read out of free text, in twelve languages, or not exist.
(Twelve, not the nine we had counted: RSNA's own announcement states twelve languages across 16 sites
on five continents. Ours was a count of what we happened to detect, not of what is there.)

And **`test.csv` has no `Report` column** — text is available when fitting and absent when predicting.
That rules out a text branch at inference and leaves reports usable only as training targets, as
auxiliary signal, or as a weight on how confidently a study could be read (`knee-6xm`, `knee-9k2`).

Read the annotated 58 as *counts*, not rates: MCL has nine positives, Lateral OA eleven, Baker's
twelve. Any decision made on the annotation check is made on single-digit positives — which is the
whole reason the holdout, not the annotation check, arbitrates (`knee-92z`).

### Three tracks, run in parallel

**1. Submission path** — the only risk that does not shrink by waiting, and it needs neither local data
nor a model. Probe kernel (`knee-61q`) → settle notebook-only vs CSV (`knee-kca`) → licence and
external-data check (`knee-0s6`) → graft (`knee-exd`).

**2. Data layer** — gated on the 247 GB download. Slot routing with the acquisition axes recovered from
the header (`knee-64b`), geometric slice ordering (`knee-7ve`), laterality normalisation with recovery
(`knee-e5e`), physical-scale crop and resample (`knee-b0c`), decode-once cache (`knee-37l`).

**3. Labels and model** — labels are the ceiling on a weak-supervision task (`knee-6xm`), then the
per-diagnosis attention head (`knee-dh7`), encoder adaptation (`knee-3om`), leak-safe splits
(`knee-92z`).

### The four design decisions, and the constraint that proves each

| decision | why | what proves it |
|---|---|---|
| Recover `Fluid_Sensitive` / `Fat_Suppression` from TR/TE/`ScanningSequence` rather than trusting the shipped columns | the two columns are reported to agree on every row, i.e. one axis where the names claim two | agreement rate of the two columns on our `train_series.csv` — 100% confirms the degeneracy |
| Order slices by `k = p·n̂` from `ImagePositionPatient` / `ImageOrientationPatient` | file names are SOP Instance UIDs: unique, not ordered, and the failure is silent | Spearman ρ of filename order vs geometric `k` — expect ≈ 0 |
| Crop to a fixed physical extent, resize so pitch ≤ d/2 | a 1 mm tear needs ≤ 0.5 mm/px; below that no downstream capacity recovers it, because it was destroyed before the first convolution | FOV distribution (`Rows × PixelSpacing`) on our corpus; the crop must sit below almost every series' FOV or it silently no-ops |
| Split by hash of the report text | byte-identical reports across studies share one derived target vector; splitting a group across the divide scores a model on a target whose source it trained on | duplicate-group sizes on our corpus; holdout AUC must not move when groups are kept whole vs split |

### Stage gates

Rigor is staged. At the finding stage a method tweak gets the **cheapest single test that shows signal**,
warm-started where the input distribution is unchanged. Sweeps, ablations, multi-seed and matched
controls are *perfecting* tools — they harden a number that already moves, and spending them earlier
buys wall-clock, not signal.

Cost-tiered: a cheap proxy gates the expensive commit, and the slow confirmation runs async.

## Phase 3 — what the second audit found

The first list of next moves was written from memory and got the headline wrong: it blamed the
per-member gap on training length. The curves say otherwise — five folds peak at epochs 13/13/10/12/11
and gain ~0 over their last five, so **training is saturated at ~12 epochs** and running longer at a
constant learning rate buys nothing. Reading their code found the actual differences:

| difference | ours | theirs | issue |
|---|---|---|---|
| learning-rate schedule | **none** (constant) | `OneCycleLR`, `pct_start=0.15` | `knee-2fk` |
| weight for an unmentioned finding | `conf` (so 0.0 — dropped) | `0.25 + 0.75·conf` (floor) | `knee-9sy` |
| loss normalisation | `/ w.sum()` | `/ element count` | `knee-sbr` |
| inference windows | 4 disjoint | 10 overlapping | `knee-36y` |

A constant LR that plateaus at 12 while a one-cycle run improves to 60 is exactly the expected shape,
which makes the schedule the first thing to test and the most likely single cause.

**The diagnostic that outranks all of them** (`knee-o42`): both medial/lateral pairs favour medial
(−0.043 meniscus, −0.066 OA), and the side-defined targets are precisely what a broken laterality
normalisation would damage. It is cheap to settle because it needs **headers, not pixels** — split the
holdout by how each study's side was decided (tag / geometry / unresolved) and compare the asymmetry.
A flat result kills the lead honestly; a worse geometry half is a bug worth more than any ensemble.

Ranked deliberately: the schedule and the diagnostic first, **members last** — 20 members is the most
predictable gain available and the least informative, and it would raise the number while leaving every
open question exactly where it is.

## The measured noise floor changes what is worth running (2026-08-11)

Three seeds of one recipe on one split spread **0.0062**. Every knob-sized change this project has left
is therefore invisible at n=1 — intensity augmentation, the weight floor, the label table and per-slice
max all measured inside it, which means *no measured effect*, not *a small one*.

Two rules follow, and they are the plan now:

1. **Run only levers that could plausibly clear 0.006 on one fold.** Encoder size, epoch budget,
   training-set size, resolution. Not head tweaks, not loss tweaks, not TTA variants.

   **Both of the first two now clear it, measured.** DINOv2-base over small is **+0.0085** (0.8166 vs
   0.8081), and 8 → 24 epochs on small is **+0.0068** (0.8149, in flight). Every knob-sized change ever
   tried here landed inside the noise; both things that cleared it are of the *more capacity, more
   training* kind. That is the shape of this problem and the plan should follow it.
2. **Anything to be KEPT needs three seeds** — about twelve minutes at four minutes a run. Budget it
   up front rather than discovering afterwards that an arm cannot be read.

The corollary drawn for ensembling — *"members two and three buy +0.0018 and +0.0007, the best single
member beats the three-member ensemble, stacking is only an Efficiency-Prize liability"* — **is
overturned and kept here only so it is not re-derived.** It was measured on a pool whose members varied
by knobs. On the BOARD, a two-member all-ours fusion scored **0.919** against 0.898 for twenty-four mixed
members, and the four-member `chosen` entry reached the best holdout this project has (0.8612) against
0.8523 for its own best member. What is flat is the curve over members that resemble each other; the
lever is pool COMPOSITION (`knee-k5kk`, `knee-nas5`), not member count.

## Open questions

- ~~**Is submission notebook-only?**~~ (`knee-kca`) ANSWERED by doing it: the kit-as-dataset design has
  now scored repeatedly, most recently 0.919.
- **Which wheels are missing** from Kaggle's image (pydicom? timm?) (`knee-61q`). Probe, don't guess:
  pushes are unlimited, submissions are 5/day.
- **Efficiency Prize** (`knee-5bv`) — MEASURED: all 905 teams are ranked automatically, the best
  accuracy on the board (0.940) sits at efficiency rank 638, and teams at the 0.888 we then held sat at
  ranks 29–50. Our best entry IS a two-member one (0.919), so the cheap entry and the accurate one are
  currently the same submission. The remaining
  unknown is whether the board scores a team's *selected* submissions, in which case a cheap entry
  competes with the accurate one instead of adding to it.
- **DINOv2 vs medical-pretrained** (`knee-zkz`) — the field defaulted to DINOv2 with no visible
  evidence against RadImageNet / MedicalNet / BiomedCLIP. Cheap, uncrowded, needs a harness first.
- **Can we beat the shared LLM label tables?** Everyone mounts the same ones, so this is the one axis
  where an improvement is not also the field's improvement.

## Assets we already have

- `comp/` — Kaggle submission machinery copied from `biohub_kaggle` and retargeted (four constants).
  Copied, not extracted: the second consumer is what reveals what is genuinely competition-agnostic.
- `../cardiac-seg` — DICOM ingest (`core/data/static/mri/dicom.py`), per-dataset source registry, store
  build/normalise/query, locked test sets. The closest existing code to this task's data layer.
- `../biohub_kaggle` — the graft pattern (mount a competitor's trained weights as a dataset), and the
  kernel gotchas now recorded in `comp/README.md`.

## Where the constraint sits now (2026-08-11)

**Supervision-limited, not capacity-limited.** Capacity was the only axis that ever paid here, and it
has stopped: base → large is +0.0021 for 3.5x the parameters, and large *peaks at epoch 10* then gives
back 0.009 while small and base peak at 13-15 and hold. A model that overfits a third of the way
through has more capacity than the labels can constrain — and 98.7% of those labels are read out of
report text.

That reading is what orders the current queue, and each item is falsifiable:

| running | why it should pay, if the reading is right |
|---|---|
| slice coverage 12 → 20 | adds EVIDENCE, not parameters: 89% of series have more than 12 slices inside the 0.20-0.80 band we chose to read, so a third of it is discarded |
| report supervision | 12 numbers → the whole report (median 129 words, twelve languages). More supervision per study, at zero inference cost |
| full-data fit | 4,407 studies against 3,527 — more supervision, not more model |
| encoder rate bracket | should be **neutral to negative**. If a faster encoder helps materially, this whole reading is wrong and the encoder was simply frozen too hard |

**What is closed.** Label *filtering* (refuted on two matched seeds, −0.0065), the anatomical prior
(neutral), EMA (+0.0010), encoder depth at the inherited rate (uninterpretable — see the bracket),
fold ensembling (flat member curve, and the owner has ruled out folds).

**What is NOT ruled out.** A better report *reader* — that resolves uncertain rows rather than striking
them, which is the opposite operation to the one refuted.
