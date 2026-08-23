# The discordant contrast is not a proxy for the metric

`side_dense` (2026-08-22, `knee-6u12`) asked the loss to order the two columns of a compartment pair
within a study, on the argument that three of four such columns resolve their own side and the fourth
sits at chance on every member this project has ever fitted. The term fired, the contrast moved the way
the arm predicted, and the score fell on every one of the twelve labels including the one the term
exists to fix. That combination is the finding, and it is worth more than the arm cost.

## What was measured

`band_dense`'s invocation with `--side-weight 1.0` appended and nothing else touched, so the control was
already on disk and no control-side variable could drift. Read with
`python -m kneemri.evaluation.side_blindness` over the same 909 holdout studies and the same n.

| column | control | `side_dense` | delta |
| --- | --- | --- | --- |
| Lateral OA (contrast) | 0.5141 | 0.5759 | **+0.062** |
| Medial OA (contrast) | 0.7416 | 0.7962 | +0.055 |
| Lateral Meniscus (contrast) | 0.7319 | 0.7192 | −0.013 |
| Medial Meniscus (contrast) | 0.8774 | 0.8066 | −0.071 |
| **holdout mean AUC** | **0.8578** | **0.7937** | **−0.0641** |
| Lateral OA, per-label AUC | — | — | **−0.0342** |

The term was verified landed and firing three separate ways before the run — `side_weight: 1.0` stamped
in the checkpoint recipe, a smoke loss of 1.9368 against the control's ~1.2 (the extra ~0.69 is exactly
`softplus(0)`, what a side-blind pair must pay), and VRAM and step time byte-identical to the control, as
a term that adds no parameter must be. This is not a null wearing a mechanism's clothes.

## The finding

**Side resolution and per-label AUC moved in OPPOSITE directions on the same column in the same run.**
Lateral OA became measurably more side-dependent (+0.062 on a contrast whose null is 0.50 by identity)
and measurably worse at the thing the board scores (−0.0342).

The mechanism is not subtle once stated. The competition metric is twelve ROC AUCs, and an AUC ranks
**study against study within one label**. The side term orders **label against label within one study**.
Those are orthogonal orderings over the same logits, and this run is the demonstration that buying one
costs the other rather than composing with it. `Objective.pairs` was never missing the side comparison by
oversight — the side comparison is a quantity the metric does not read.

That retires the contrast as a selection instrument. It remains a valid *diagnostic* — the observation it
produced, that Lateral OA carries no side information on any member, is untouched and still unexplained —
but a column moving on it predicts nothing about the score, and no future arm should be priced on it.

## What it was NOT

Not the pre-registered mass-imbalance failure. The arm predicted that pooling every pair before dividing
would spend the gradient 2.6:1 on the meniscus pair, lifting the pair that already resolves at the OA
pair's expense. The meniscus pair did not gain; it lost too (Medial Meniscus −0.071). Damage is uniform
across labels the term places no gradient on at all — Baker's −0.207, MCL −0.077, Fracture −0.025 — which
is shared-encoder damage, not pair-specific robbery. Per-pair normalisation therefore fixes nothing here;
it redistributes a cost rather than removing it.

Not a harness failure. Plumbing verified three ways above; data and split identical by construction;
training differs by one flag from a control already on disk; the test read both arms with one tool on one
set of studies. The link that broke is the **idea, in this specific use**.

## A caveat the tool now carries

The neutral control column went dirty under the arm. `PF OA` belongs to neither side, and
`side_blindness` uses it to detect the burden confound — the medial-only group being sicker knees rather
than differently-located ones. It reads 0.4770 / 0.4133 under the control (clean, straddling chance) and
0.4274 / 0.3498 under the arm, which the tool flags CONFOUNDED in both pairs.

The term puts literally zero gradient on `PF OA` (asserted in
`tests/unit/kneemri/training/side_ranking.py`), so this is indirect coupling through the shared encoder:
pushing two columns of the trunk apart moves a third that reads the same features. The consequence is a
limit on the instrument. **`side_blindness` can audit a run trained WITHOUT a side term; it cannot cleanly
audit one trained with it**, because the term dirties the control the tool depends on. That is also a
second, independent reason the contrast cannot be used to steer a side-term weight sweep: the sweep would
be read on a ruler the sweep itself bends.

## What this does not rule out

- **The side-blindness observation stands.** Lateral OA sits at chance on all eight members read
  separately, across recipes varying the loss, the schedule, the augmentation and the input pixels.
  `knee-yrg` separately refuted four data-side explanations. Nothing here explains it.
- **The side IS recoverable from this representation.** The contrast rose 0.062 on demand, so the
  information is present and reachable. What is refuted is getting it out of the loss for free.
- **The term survives as an asset at low weight on a head that sees position.** The live suspect for OA
  specifically is now the pooling stage or the field of view for those compartments; a side term is
  re-testable as a small regulariser on top of such a head, and would be read on per-label AUC rather
  than on the contrast.

## The follow-up that was declined, and why that is the arm's return

The pre-registration named weight 0.3 as the PARTIAL branch's fix. It is not being run. The result says
the **sign** of the trade is wrong, not merely its magnitude, so a third of the weight buys a third of a
bad trade. Priced out loud: 73 minutes of card, P(clears 1.5% on the mean) ≈ 0.02. Declined, along with
per-pair normalisation, for the reason above.

An arm bought as an instrument returned an instrument's answer: it closed a whole axis rather than one
setting of it, and it retired a ruler that would otherwise have been quoted at every future arm on this
question.
