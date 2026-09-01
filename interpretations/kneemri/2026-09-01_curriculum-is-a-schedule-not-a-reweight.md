# Curriculum is a SCHEDULE, not a reweight

2026-09-01. Read off three arms that were already on disk, plus one control. No card time was spent to
produce this; the card time had been spent weeks earlier and the arms had never been read against each
other.

## The 2x2

Difficulty-aware training has two independent choices here: WHICH signal says a study is hard, and WHEN
the intervention is applied. Three of the four cells were already measured, at 12 epochs, on one recipe,
against one holdout.

| | static (from step one) | schedule (over the first half) |
| --- | --- | --- |
| **label certainty** | — | `certainty_ramp` **0.8584** |
| **member disagreement** | `disagreed_full` 0.8471 | *empty* |

Controls: `disagreed_uniform` **0.8507** is the matched alpha-zero control for the disagreement row — it
matters that it is not `Uniform`, because `Disagreed` draws WITH REPLACEMENT and at alpha zero is a
bootstrap resample rather than a once-through epoch. `band_dense` **0.8578** is the ledger's uniform
draw and the control for the certainty row.

Holdout noise floor is 0.0062.

## What it says

- The static tilt **costs 0.0036** against its own control. Inside the floor, so it is FLAT-with-a-sign
  rather than a measured loss — but it is unambiguously not a gain.
- The schedule **pays 0.0077** against its own control. Above the floor.

Both are supervision-axis changes on the same recipe and the same corpus. What separates them is not
which studies are treated as hard; it is whether the treatment arrives before the model can use it.

## Why the mechanism is not post-hoc

`sampling.py` already records the reason the weight is disagreement and not loss: ~87% of the residual's
mass is aleatoric, so a loss-ranked draw mines label noise. And `residual_split` puts ~70% of the pool's
errors in the band that most-but-not-all members get wrong. Those two facts say the disagreement signal
points at real structure. They do not say a model at initialisation can use it — a network that has not
learned the easy structure yet can do nothing with a hard study except memorise it, and a tilt applied
from step one spends its entire first pass there.

That is the failure a curriculum exists to prevent, and it is what makes `disagreed_full`'s loss
informative rather than a dead end.

## Precedent

Zhou et al., ISBI 2023 ([PMC10602195](https://pmc.ncbi.nlm.nih.gov/articles/PMC10602195/)):
elbow-fracture radiographs, 1,865 patients, VoG-ordered easy-to-hard curriculum, binary AUC
0.8407 → 0.8652 (**+0.0245**), multi-class +0.0201, with no human difficulty annotation. Surfaced by
Kimi's Q-S review of the supervision axis, where it was the only intervention with a published AUC delta
on a medical classification task. The shape we measured is the same shape; our size is a third of theirs,
which is what an adjacency discount looks like when it is honest.

## What is being run because of it

`disagreed_ramp` (bead knee-zer5) fills the empty cell: alpha 1.0, the `disagreed_full` destination,
reached over 6 of 12 epochs, the `certainty_ramp` schedule length. Both controls are on disk, so no
control arm runs. Branches are pre-registered in the arm header — the one worth naming here is the
refutation: at or below 0.8471 the disagreement signal is dead at any schedule, both cells of that row
carry a negative, and the difficulty axis closes with certainty as its only usable signal.

## What it does NOT say

It does not price the lane. +0.0077 on one member is a member-quality gain, and the champion's own
record prices that lane as ceilinged near 0.943 against a leader at 0.952 — improving all six seats at
once on the strongest measured axis moved the board +0.002. So a curriculum member is worth a seat and
is not worth a plan. The route this project still has to find is elsewhere, which is why the letter that
went out the same hour asks about external corpora rather than about another supervision knob.

It also does not establish that the ramp LENGTH is right. Six of twelve was taken from the one schedule
that has paid here, not fitted. Fitting it would be a knob sweep on a knob-sized gain and is not
licensed by anything above.
