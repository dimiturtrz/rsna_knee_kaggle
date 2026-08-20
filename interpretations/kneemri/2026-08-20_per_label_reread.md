# Re-reading eleven vision arms per label: the settled recipe is defended by two labels

*2026-08-20. Free re-read — no card time. `kneemri.evaluation.run_comparison --control
band_dense_s2026.pt`, logs in `logs/vision_reread.log` and `logs/vision_reread2.log`.*

Every vision/decode arm with a surviving checkpoint, re-marked against the recipe the project actually
ships (`band_dense`, 0.8578 at epoch 8) instead of against whatever control it was written for. Eleven
arms × twelve labels = 132 cells. This is a re-read of numbers already banked, so it costs nothing and
can only buy structure, never a new result.

## The load-bearing observation: PF OA is negative in 11 of 11

| arm | MEAN | PF OA | Synovitis |
|---|---|---|---|
| `band_wide` | −0.0025 | **−0.0194\*** | −0.0135 |
| `band_span` | −0.0013 | **−0.0164\*** | **−0.0194\*** |
| `band_density` | −0.0038 | **−0.0267\*** | **−0.0358\*** |
| `band_full` | −0.0151 | **−0.0382\*** | **−0.0275\*** |
| `depth_regions` | −0.0029 | **−0.0193\*** | −0.0036 |
| `depth_regions_span` | −0.0091 | **−0.0212\*** | −0.0024 |
| `attend_dense` | −0.0009 | −0.0080 | +0.0002 |
| `aug_long` | −0.0067 | −0.0100 | **−0.0181\*** |
| `crop80_span` | −0.0112 | **−0.0302\*** | −0.0147 |
| `crop80_tissue` | −0.0233 | **−0.0370\*** | **−0.0207\*** |
| `centre_tissue` | −0.0169 | **−0.0438\*** | −0.0068 |

PF OA: **11/11 negative**, seven of them clearing their own pair's paired half-width. Synovitis:
**10/11 negative** (the eleventh is +0.0002, i.e. zero). MEAN: 11/11 negative.

These arms are not independent — they share a control and several share recipe axes — so the naive
binomial (p ≈ 0.001 for 11/11) overstates it. What survives that caveat is the *shape*: two labels
move against every single perturbation of the settled operating point, across levers that have nothing
in common with each other (band span, band density, slice pooling, region pooling, crop geometry,
augmentation schedule, epoch count).

**Reading.** `band_dense` sits at a local optimum that PF OA and Synovitis defend. That is consistent
with what the band square already said — the 0.10–0.90 span is exactly what *resolved* those two rows
(`band_dense` gained PF OA +0.0192\*, Synovitis +0.0218\*) — and it has a cost this ledger has been
paying without naming: **the mean ruler penalises exploration**, because every move away from the
optimum surrenders the two rows the optimum was chosen for, and those losses are large enough
(−0.02 to −0.04) to sink a mean that is otherwise flat or positive elsewhere. `band_span` is the
cleanest case: focal −0.0036, regional +0.0019, and a MEAN of −0.0013 that is almost entirely two
rows.

## The positive cells are NOT a finding

Three cells clear their pair's half-width on the positive side:

- `band_density` Lateral Meniscus **+0.0187\***
- `depth_regions` Lateral Meniscus **+0.0165\***
- `aug_long` ACL **+0.0187\***

132 cells at a two-sided ~5% marking rule predicts **≈3.3 starred cells per side by chance alone**.
Three is what chance predicts. Neither the Lateral Meniscus pair nor the ACL cell is evidence of
anything on this read; they are directional at most, and the honest statement is that this re-read
found **no per-label gain that survives its own multiplicity**. Lateral Meniscus is positive in only
4 of 11 arms, so it does not have the sign consistency PF OA does either.

The asymmetry between the two sides is itself the finding: the negative side is *structured* (one
label, every arm), the positive side is *scattered* (no label repeats with consistency). Structure on
one side and noise on the other is what a defended local optimum looks like.

## Three arms could not be re-read

`resolution_420_s2026.pt`, `extended_slots_s2026.pt` and `multiscale_s2026.pt` report
`NOT READ: does not exist` — the checkpoints are gone from the data root. Their ledger rows stand on
their original marking and cannot be re-marked against the shipped recipe. That is a coverage gap in
this document, stated rather than papered over.

## Prior art check: per-label combination already exists here

The obvious next move from a table like this — pick members per label — is **already implemented**, in
`kneemri/evaluation/label_combine.py`, and it was built around exactly the trap this table would walk
into. Its own docstring:

> Choosing members per label on the holdout that then reports the number is selection optimism … So
> the deliverable is not the combine — it is the combine WITH an estimator that fits the choice on
> studies it is not scored on.

So "roster per label" is prior art in this repo, complete with a split-half estimator and a named
un-shippable ceiling (`member_curve.oracle`). Nothing in this re-read is a reason to rebuild it, and
the three chance-level positive cells are certainly not a reason to *feed* it.

## What this changes

Nothing about what to run next, which is the honest size of a free re-read. What it changes is how to
*read* a flat mean on this recipe: a MEAN of −0.001 to −0.004 on an arm whose PF OA row is −0.02 is
not "the lever did nothing", it is "the lever paid the local-optimum tax and broke even on the other
ten labels". Arms whose mechanism is not about the central compartment should be marked on the other
ten rows before their mean is believed — and `band_dense` itself should not be treated as a neutral
control for anything that moves the band.
