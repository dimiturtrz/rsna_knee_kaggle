# kneemri — PLAN (source of truth)

This is the canonical plan. Read it at session start before proposing work. Substance lives here;
ordered execution status lives in `ROADMAP.md`. Work items live in beads (`bd ready`), and this file
cites them by id.

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
| mean of 12 AUCs (LB) | 0.5 (constant submission) | > public plateau ~0.899 | — |
| worst single-label AUC | 0.5 | — | — |
| submission path works end to end | — | proven | — |

Track the **worst label**, not only the mean — that is where the 1/12 is being forfeited.

## Approach

Public evidence → our-data statistics → method, in that order. The public evidence is read for
**mechanism**, and every borrowed constant is re-derived on our own corpus before it is trusted
(`knee-25n`); a constant that survives our measurement is a finding, one that does not is a bug we
would otherwise have inherited.

### What the task actually is

**Measured 2026-08-10: 4407 training studies, all 4407 carry a `Report`, and exactly 58 carry all
twelve annotations.** 1.3%. This is a weak-supervision problem first and an imaging problem second —
the labels for 98.7% of the corpus have to be read out of free text, in nine languages, or not exist.

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

## Open questions

- **Is submission notebook-only?** (`knee-kca`) The entire kit-as-dataset design in `comp/` is contingent
  on it. Unverified — the overview page is JS-rendered.
- **Which wheels are missing** from Kaggle's image (pydicom? timm?) (`knee-61q`). Probe, don't guess:
  pushes are unlimited, submissions are 5/day.
- **Efficiency Prize** (`knee-5bv`) — a separate leaderboard. Target it deliberately or not at all.
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
