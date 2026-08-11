# Submissions

Kernels are named for the METHOD, not for the attempt: `kneemri-<source>-<variation>`. A new version of
one slug is that same method retrained or re-tuned; a different source or aggregation is a different
slug. The slug is the claim, so a grafted score can never quietly become ours.

Budget: **5 submissions per day**. Kernel *pushes* are unlimited — probe there, submit only when the
question needs the leaderboard to answer it.

## Ledger

| date | kernel | v | method | local | LB | note |
|---|---|---|---|---|---|---|
| 2026-08-10 | `kneemri-probe` | 2 | passthrough `sample_submission` | — | not submitted | environment + corpus probe |
| 2026-08-10 | `kneemri-graft-verify` | 6 | fingerprint check only | 20/20 @ 4.65e-06 | not submitted | proves the architecture port |
| 2026-08-10 | `kneemri-graft` | 1 | **GRAFT** — 20 published CC0 members, our data layer, rank-averaged | their holdout 0.8398 | **0.888** | **their result, not ours** |
| 2026-08-10 | `kneemri-trained` | 1 | **OURS** — 5 folds fitted by `kneemri.train` on report-derived labels | holdout mean 0.8162 | **0.868** | trained with augmentation silently disabled |
| 2026-08-11 | `kneemri-trained` | 3 | **OURS v2** — augmentation actually applied + OneCycle + overlap TTA | holdout mean **0.8269** | **0.888** | **equals the 20-member graft with 5 of our own** |

## What counts as ours

The graft row is the field's floor measured on our harness. It is **not a result**: the weights are
`pilkwang/rsna-knee-weights` (CC0), and the only thing of ours in it is the data layer and the ensemble.
It exists so that every later number has something to be compared against, and so the submission path
was proven before a model depended on it.

`kneemri-trained` is ours end to end — our slot routing, our ordering, our laterality, our crop, our
head, our fitting. It starts below the graft, which is the honest starting position for a replication
plus one day's work.

## The two numbers, side by side

| | members | epochs each | per-member holdout | public LB | rank of 1061 |
|---|---|---|---|---|---|
| graft (their weights) | 20 (5 folds x 4 seeds) | 60 | 0.8398 | 0.888 | 349 |
| ours (one day) | 5 (5 folds) | 16 | 0.8162 | 0.868 | 393 |

**0.020 apart, from an ensemble four times smaller trained for a quarter as long.** The public median
is 0.819, so both clear it; the leader is 0.942, so neither is close. Rank 349 with the field's own
weights is the tell that a graft scores where forks score — 86 teams sit within 0.005 of it.

The comparison is only this clean because both sides train on the SAME public label tables and run
through the SAME data layer. Members and epochs are the two axes we have not spent, and they are cheap:
27 s per epoch on the 5090 over the memory-mapped cache. Spending them is the obvious next move and
also the LEAST informative one — it buys rank without teaching anything. The labels are where the
un-shared edge is.

## What 0.888 bought

The graft is not only a floor — it **prices our data layer in leaderboard units**. The weights are the
same CC0 members the public forks run, so the difference between our 0.888 and what those weights score
in other hands is our pipeline's reading of the pixels.

**Correction to the first reading of this.** 0.899 was taken from a public notebook's title and called
"the plateau"; on the actual leaderboard 0.899 is about 100th place and the frontier is 0.942. So the
0.011 gap is to 100th place, not to the people running these weights best, and the real distance to the
top is 0.054 — far too large to be the slot tie-break. Something structural separates the leaders from
the public stack.

That agrees in direction and rough size with the 0.025 out-of-fold residual measured locally, which
promotes the slot tie-break from a shrug to a lead: when two candidate series have equal slice counts,
a stable `sort_values` picks the first in FRAME order and our `max()` picks the first in LIST order, and
a different series in a slot is a different image for the head to read.

The measurement that would settle it: re-run the out-of-fold oracle logging which series filled each
slot, and count how many studies have a tie at all. If ties are rare and the residual survives, the
cause is somewhere else and this lead is dead — which is worth knowing just as much.

## v2: our five members equal their twenty

| | members | public LB | rank of 1077 |
|---|---|---|---|
| graft (their weights) | 20 | 0.888 | 367 |
| ours v1 | 5 | 0.868 | 412 |
| **ours v2** | **5** | **0.888** | **367** |

**+0.020 on the leaderboard, 45 places, from three defects in our own code** — augmentation never
called, no learning-rate schedule, and the TTA default left off. No new idea, no bigger model, no
extra data.

The graft has since stopped being a floor and become a *control*: same data layer, same ensembling,
same submission path, differing only in whose weights and how many. Matching it with a quarter of the
members says our per-member training is now roughly on par with the published one — which is what the
holdout was already saying (0.8269 against their reported 0.8398, on different splits) and is now
confirmed on the hidden test, where neither of us chose the studies.

The remaining 0.054 to the leader is untouched by any of this, and nothing measured so far explains
it. It is not the schedule, the augmentation, the labels, the slice count or the TTA.

## What the ablations say (fold 0, matched; ruler held at pilkwang)

|  | pilkwang labels | v4_blend labels |
|---|---|---|
| geometric augmentation | **0.8202** | 0.8194 |
| + intensity augmentation | **0.8221** | 0.8203 |

Both imported ideas are null: intensity augmentation +0.0014 on average, better labels −0.0013, no
interaction. Set against what fixing our own code was worth — augmentation never called +0.0110, the
missing schedule +0.0061, the TTA default +0.0035 — **repairing what we had beat importing what others
had by an order of magnitude.**

That is not an argument against reading the field: the reading is what found the schedule, and the
`.903` notebook is what made label quality checkable at all. It is an argument against assuming a
published gain transfers. Every imported idea here either needed adapting or died on a mechanism
specific to this architecture — channel-stacked slices, side-defined targets, a ViT that is not
translation-equivariant, readers that are not independent.

Intensity augmentation stays behind `--intensity-aug` rather than being deleted: the holdout is drawn
from the same corpus as training, so it is structurally blind to the cross-scanner axis that transform
exists for. It is worth one submission slot when a slot is being spent anyway — a claim that is only
honest because it was written down before the result.

## The two oracles

Anything claiming to reproduce the published pipeline has to pass both, because they check different
things and neither implies the other:

- **Fingerprint** (`kneemri-graft-verify`) — each member's output on a seeded synthetic bag. Proves the
  ARCHITECTURE is the function those weights were fitted to. Blind to the pixels: it feeds noise.
- **Out-of-fold agreement** (`scratchpad/oof_oracle.py`) — our pipeline against the member's own stored
  predictions on the studies it held out. Proves the PIXELS: routing, order, laterality, crop,
  percentile, resize. Currently max|diff| 0.025, per-study correlation 0.99, under disjoint windows.

The residual 0.025 is an OPEN QUESTION, not a rounding error — a faithful pipeline would agree to about
1e-6, as the fingerprint does. The likeliest cause is the slot tie-break when two candidate series have
equal slice counts.

## Gotchas paid for in full

- **Submission is notebook-only.** `kaggle competitions submit -f submission.csv` returns 400. The
  working form is `-k <owner>/<kernel> -v <version> -f submission.csv`.
- **Search for the artefact, not the container.** Three kernel runs died on mount-shape assumptions:
  the competition sits under `/kaggle/input/competitions/<slug>`, datasets under
  `/kaggle/input/datasets/<owner>/<name>`, and a `--dir-mode zip` upload is expanded on mount with its
  wrapping directory REMOVED.
- **Probe what you IMPORT, not only what you plan to use.** The runtime probe checked pydicom, timm and
  transformers (all present) and never checked `jaxtyping`, which our own modules import at module top.
- **A published checkpoint is a training artefact**, not a bare state dict: weights sit under `model`
  beside epoch, fold scores, config, out-of-fold predictions and its own fingerprint.
- **`machine_shape: NvidiaTeslaT4` gives two T4s and only four CPUs.** On a decode-bound pipeline the
  CPU count binds first: measured 12 ms per DICOM slice off the mount.
