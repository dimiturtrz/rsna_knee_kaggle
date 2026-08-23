# The per-label deficit axis is spent: everything it could buy is +0.0043

*2026-08-23. CPU-only arithmetic over predictions already on disk. New instrument:
`kneemri/evaluation/weights_gap.py`, log in `logs/weights_gap.log`. No card.*

## What was measured

`published_ruler` states in its own docstring that it "answers questions about combining, not about
whose weights are better", and that a comparison between ours and theirs "still belongs on the shared
fold-0 holdout". Nothing had built that comparison. Two per-label profiles were therefore in the ledger,
disagreeing about which label is cheap, differing in **three** things at once — studies, labels, weights:

| ruler | weights | studies | weakest label |
|---|---|---|---|
| corpus (`2026-08-14`) | theirs, 20 published CC0 members | 4,407 | Lateral Meniscus 0.7981 |
| fold-0 | ours, `band_dense_llrd` | 909 | MCL 0.7422 |

`weights_gap` removes two of the three. Their five folds and our one were drawn independently, so no
single published member covers our holdout — but within one seed their five folds partition the corpus
exactly, so stacking a seed gives one held-out prediction per study from one model. Four seeds, four
single-member columns, restricted to our 909 studies and scored against the same graded truth at the
same midpoint by the same rank AUC. Members whose `oof` covers more than 40% of the corpus are refused:
their `fold: -1` members trained on everything they recorded.

## The table

Our single member against their average single member, same 909 studies:

| label | ours | their 1 member | their 4-seed pool | ours − their member | n_pos |
|---|---|---|---|---|---|
| **MCL** | 0.7423 | 0.7769 | 0.7889 | **−0.0346** | 171 |
| **Fracture** | 0.7827 | 0.7994 | 0.8102 | **−0.0167** | 68 |
| Medial OA | 0.8702 | 0.8633 | 0.8680 | +0.0069 | 333 |
| Lateral OA | 0.8099 | 0.8017 | 0.8079 | +0.0082 | 233 |
| Synovitis | 0.8717 | 0.8576 | 0.8661 | +0.0141 | 124 |
| Effusion | 0.8533 | 0.8387 | 0.8451 | +0.0146 | 558 |
| Contusion | 0.8700 | 0.8447 | 0.8524 | +0.0253 | 267 |
| ACL | 0.8784 | 0.8493 | 0.8573 | +0.0292 | 248 |
| Baker's | 0.9078 | 0.8760 | 0.8866 | +0.0319 | 217 |
| PF OA | 0.8544 | 0.8197 | 0.8262 | +0.0348 | 409 |
| Medial Meniscus | 0.8975 | 0.8445 | 0.8531 | +0.0529 | 482 |
| Lateral Meniscus | 0.8419 | 0.7708 | 0.7808 | +0.0712 | 226 |
| **MEAN** | **0.8483** | **0.8285** | **0.8369** | **+0.0198** | |

## The finding, and it closes an axis rather than opening one

**Our single member beats their single member on ten of twelve labels, and beats their four-seed pool
on the mean.** The only labels where somebody else's weights are demonstrably better are MCL (−0.0346)
and Fracture (−0.0167).

Each label carries 1/12 of the metric. So the **entire** demonstrated per-label headroom — closing both
deficits completely, to weights that provably exist — is

    (0.0346 + 0.0167) / 12 = +0.0043

That is a quarter of the 1.5% bar, and it is the CEILING of the whole "find our weak columns and fix
them" programme, not an estimate of what an arm would get. And Fracture's share of it is not real: 68
positives in this holdout carries a standard error near 0.04, which is `label_power`'s prediction and
the reason the ledger already retired Fracture as a target. On the corpus ruler Fracture is our
strongest label at 0.8923. The honestly-purchasable number is **MCL alone, +0.0029 on the mean.**

## What this does to the work in flight

- **`focus_latmen` was correctly killed, for a better reason than the one given.** It was built to chase
  Lateral Meniscus because the corpus ruler calls it the weakest label. It is the label we beat their
  weights on by the **largest** margin in the table (+0.0712). Their profile was their model's problem.
- **`focus_mcl` is cancelled before it runs.** It is a 3.1 h arm whose *complete success* — closing a
  gap another model has demonstrated — is worth +0.0029 on the mean, five times under the bar. The
  mechanism it tests (does jointness cost a column?) is sound and untested, and that does not save it:
  the prize is too small to pay for the card whatever the answer is. The arm file stays in the tree with
  this document named in its header, so a future reason to run it re-tests it rather than assumes it.
- **`knee-gisc` (the MCL recipe-axis ceiling) is downgraded, not closed.** Its measurement stands — MCL
  does not move for crop, slices, resolution, encoder family or slot scheme. What changes is that MCL
  was never worth the arms it was going to get.

## What it does NOT rule out

- **It is not a statement about the leaderboard.** Their published members sit near 0.899 on the board
  and the frontier is 0.952; ours score 0.934. This table says our per-member weight quality is ahead of
  theirs on ten of twelve labels on our own report-label ruler. `ruler_transfer` and `board_transfer`
  are the modules that price how much of that survives the change of ruler, and neither has been re-read
  against this table.
- **It says nothing about combining**, which is what `published_ruler` was built for and what carries a
  public entry from one member at 0.83 to a pool near 0.899. Their pool gains +0.0084 over their single
  member on the mean here. Ensembling remains a live axis; per-label weight quality does not.
- **It does not price a genuinely different encoder**, only different weights of the same broad family
  fitted on this corpus. An architecture that sees MCL better than anything here would still show up as
  a gap this instrument cannot anticipate.

## Where the 0.018 to the frontier is not

Three candidate locations for the gap have now each been priced and found too small: per-label deficits
(+0.0043, here), the recipe axis around `band_dense` (knob-sized by construction, `experiments/README.md`),
and subset selection over the existing member pool (selection overfit +0.0033, `2026-08-23_selection-overfit`).

**And the obvious fourth is already spent — this section said otherwise in its first draft and the ledger
corrected it within the hour.** Data volume and the input itself (17.4% of series never routed, a tenth of
the acquired slices used, 336 px for few-voxel findings) is `knee-jk3`'s axis, and all four of its
starvations were measured by 2026-08-21: whole-stack `+0.0147` SHIPPED, extended slots `+0.0002`, twenty
cached slices `−0.0001`, 392 px `+0.0002`, against a `0.0062` floor. One paid, and it paid by changing
WHICH data a step sees rather than by adding any. `depth_true` then closed the order axis that could have
made those nulls unreadable (0.8461, seat `−0.0002`). Writing "the only unpriced axis" over a bead whose
own notes say "PRESCRIPTION SPENT" is exactly the failure the read-the-ledger rule exists for.

So this document prices an axis closed and names no replacement. What the ledger leaves genuinely open is
narrower than any of the above: a 3D encoder that sees the volume BEFORE per-slice features exist (which
`depth_true`'s close explicitly does not rule out), and the label/loss side, which is blocked behind a
58-study vision ruler (`knee-7d1`) rather than measured. Both are stated as open by the beads that closed
their neighbours, and neither has a priced mechanism yet. That is the honest state, and inventing a fourth
priced axis to end the document on is how the first draft got it wrong.
