# The anneal give-back is universal, small, and not a recipe property

**2026-09-17.** Read off the per-epoch lines of runs already on disk. No card was spent for it.
**This file was first written as a two-run contrast and is corrected below by the 212-run sweep that
followed twenty minutes later.** The retraction is kept in place because the wrong version had already
repriced a bead, and the shape of the error is the useful part.

## What was read first, and why it looked like a finding

Two arms, both `one_cycle`, both midpoint macro on `holdout909`, from their own `.full.log`:

| run | epochs | peak | finish | give-back | train loss, peak → finish |
|---|---|---|---|---|---|
| `calib_dense_hold_s2026` | 12 | e6 **0.8634** | e11 0.8598 | **−0.0036** | 0.6494 → 0.5130 |
| `labelwise_depth24_drop40_long_s2026` | 24 | e18 **0.8679** | e23 0.8673 | **−0.0006** | 0.6036 → 0.5600 |

Read as a pair this says the shipping recipe absorbed an overfit the dense corner suffers: six times
less held-out AUC surrendered while training loss keeps falling. The obvious attribution was
`--slot-dropout 0.40` plus the doubled epoch budget, which is exactly what the shipping arm adds.

## The attribution died first

`band_dense_drop40_long_s2026` carries **both** of those flags — 24 epochs and `--slot-dropout 0.40` —
and gives back **−0.0031**, indistinguishable from the dense corner's −0.0036. The two 12-epoch drop40
arms (`blendhold_drop40` 0.8507, `chainfour_drop40` 0.8589) both peak on their LAST epoch and give back
nothing, which is `convergence_check`'s UNDERTRAINED, not flatness.

The fallback candidate was `labelwise` pooling, the shipping arm's other distinguishing flag. It dies
the same way: `chainfour` (labelwise) gives back **−0.0067** and `chaindense` (labelwise) **−0.0049**,
both worse than the max-pooled corner the story was built to explain.

## Then the contrast itself died

Sweeping every arm on disk with a log and at least 8 recorded epochs — **n = 212**:

```
give-back   min -0.0433   p10 -0.0065   p25 -0.0033   median -0.0014   p75 -0.0003   p90 0.0000
```

`calib_dense_hold`'s −0.0036 sits at about **p24**. `labelwise_depth24_drop40_long`'s −0.0006 sits at
about **p65**. Both are ordinary members of one distribution. The six-fold ratio that looked like a
mechanism is two draws from a spread whose interquartile range is 0.0030 wide — and per-epoch holdout
AUC wobbles 0.002–0.003 between adjacent epochs in these same logs, so a 5-epoch give-back of 0.0036
is roughly one wobble. There is no two-regime story. There is one corpus-wide number.

## What survives, and it is the stronger version

**The channel is empty everywhere, not just at the shipping corner.** Held-out AUC surrendered after
the peak is the measurable proxy for what a flatness-seeking optimiser has to recover by reshaping the
late anneal. Corpus-wide that proxy has a median of **0.0014** and a p10 of 0.0065, against a paired
floor of **0.0062**. So SAM-via-decay-removal has about a quarter of the floor to win at a typical
corner and clears it only in the worst tenth of runs. That is a better-founded version of the
repricing than the two-run read gave, and it points the same way.

**None of it is a lost result.** `train.py:141` loads `kept.state` before the final save, and `kept` is
the best epoch on the holdout. Every ledger number in this repo is already a PEAK, the 0.8634 above
included. The give-back was never money left on the table; reading it as headroom would have been the
`knee-f9aao` error one level down.

**The peak lands at 0.73 of the budget, median.** p10 is 0.55, p90 is 1.00. So a quarter of the
epochs this project has ever run came after the epoch that actually shipped — a wall-clock statement,
not an AUC one. It is not straightforwardly harvestable: the p90 of 1.00 means better than a tenth of
runs peak on their final epoch and are budget-bound, so a global truncation rule would cut exactly the
arms that still had gains. It is a corpus prior for `trajectory_check`, which already owns truncation,
and it is the number to quote when an arm's budget is being chosen.

## The shape of the error, which is the part worth keeping

A two-run contrast on a statistic with a 0.0030 interquartile range was written up as a recipe
property, and it repriced a bead before anything checked whether the two numbers were unusual. The
sweep that would have caught it cost one command over logs that were already on disk. The rule this
argues for is narrow and mechanical: **a per-run statistic quoted as a difference between two runs
needs that statistic's own corpus spread in the same sentence, or it is not evidence.** This is the
board-slope lesson (`CLAUDE.md`: a slope without its residual scatter cannot say whether a delta is
evidence) arriving on a second axis, and it was not transferred until after the same mistake was made
again.

## Related

`knee-k3v5t` carries the corrected repricing. `knee-38mk3` is the sibling defect for the ruler —
ranking members read on different study sets — found in the same afternoon's sweep.
