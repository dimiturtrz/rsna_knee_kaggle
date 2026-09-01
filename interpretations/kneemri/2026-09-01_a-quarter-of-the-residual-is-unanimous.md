# A quarter of the residual is unanimous, and almost none of it is the combiner — knee-y7v3

Reproduce: `python -m kneemri.evaluation.residual_split --records <the twelve honest12 checkpoints>`.
Twelve separately-trained members, 909 studies every one of them held out, one prevalence-matched
operating point per finding (as many calls as there are positives, so the threshold is not a free
parameter). The ensemble row is the pool's own `RankAverage` vote, not a mean of probabilities.

## The reading

| finding | positives | missed | unanimous | minority |
|---|---|---|---|---|
| ACL | 248 | 66 | 14.4% | 8.3% |
| MCL | 171 | 82 | 24.4% | 6.7% |
| Medial Meniscus | 482 | 79 | 22.8% | 4.4% |
| Lateral Meniscus | 226 | 79 | 13.3% | 5.7% |
| Medial OA | 333 | 80 | 27.5% | 2.5% |
| Lateral OA | 233 | 91 | 29.7% | 5.5% |
| PF OA | 409 | 94 | 22.9% | 4.3% |
| Effusion | 558 | 96 | 29.7% | 3.1% |
| Synovitis | 124 | 58 | 25.0% | 2.6% |
| Baker's | 217 | 50 | 25.0% | 12.0% |
| Contusion | 267 | 77 | 25.3% | 1.9% |
| Fracture | 68 | 44 | **33.0%** | **14.8%** |
| **POOLED** | — | 896 | **24.3%** | ~5.4% |

`unanimous` = the share of the ensemble's mistakes that ALL TWELVE members also make at their OWN
prevalence-matched point. `minority` = the share fewer than half of them make. Errors are twice the
missed count, because FN equals FP by construction at this operating point.

## What it says

**The combiner lane is worth at most ~5.4% of our errors.** Only that share of the ensemble's mistakes
are cells where fewer than half the members were wrong — i.e. where the information to be right was in
the pool and the vote threw it away. This is an independent instrument arriving at the reading six board
rows already carried: reweight, add, swap, count and vote are all flat (raptorweight +0.001, vjepaseat
−0.001, holdswap 0.000, dark7 flat, spanseat 0.944, stackseat 0.000). The seating axis is not closed
because we ran out of seatings; it is closed because there is almost nothing seated wrongly.

**A quarter of the residual is unanimous** — no reseating, reweighting or combining rule inside this
family can touch it. That is the per-cell face of `uncertainty_split`'s 12.7%-reducible bits: both
instruments say the pool agrees far more than it disagrees, and both bound the same lane.

**The middle band is where the work is.** ~70% of errors are made by more than half but not all members.
Those are neither combiner errors nor unanimous ones: they are cells a genuinely different member could
plausibly get right while most of the current family gets it wrong. That is the same place the frozen
probe points (`2026-09-01_the-trunks-are-tied-frozen-and-not-fitted.md`): a trunk that is equal frozen and
worse fitted is a candidate for exactly this band, because its per-finding profile differs by up to 0.03
in both directions.

**Fracture is the extreme on both cuts at once.** 68 positives, 44 missed (65%), the highest unanimity in
the table at 33.0% AND the highest minority share at 14.8% — the only finding where the members both
agree hardest on being wrong and disagree hardest on the rest. The unanimous half agrees with the label
lane (`2026-08-30_label_lane_closes.md` prices Fracture's headroom at +0.0160): a sampler cannot invent
supervision that is not there. The minority half does not, and is the one cell in this table where a
combiner change has anything to win.

**Contusion's 1.9% and Medial OA's 2.5% are the monoculture end.** On those findings the twelve members
are almost never split — when the ensemble is wrong, at least half of them are wrong together. A pool with
near-zero minority disagreement on a finding is reading that finding as one model.

## What this does NOT establish

- **FN and FP are equal by construction here.** The prevalence-matched point forces it, so this table
  cannot say whether we over- or under-call — only WHICH studies and WHO agreed. A separate cut at a fixed
  threshold would answer that and is not run.
- **The unanimity share is a LOWER bound on what a different model class could fix**, for the same reason
  the epistemic term is: twelve members of one family are not an honest sample of all models.
- **`mae_matched_s2026` (0.6735 solo) is in this pool**, which raises the minority share relative to a pool
  of twelve strong members — a weak member disagrees often and is usually the one that is wrong. The ~5.4%
  should therefore be read as generous to the combiner lane, not stingy.
- **Nothing here is a per-population cut.** knee-y7v3 also asks for slice count, laterality and scanner,
  which the manifest may or may not carry; that cut is not run and the bead stays open on it.
