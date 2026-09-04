# The diversity ruler was checked against the gains it predicts, and the citation does not transfer

**Date:** 2026-09-04 · **Tool:** `python -m kneemri.evaluation.fault_ruler` · **Status:** measured, 166 members

## What prompted it

Kimi's Q-DE reply arrived carrying a direct attack on the statistic this project selects members on.
Pokhrel et al. (Frontiers Digit. Health 2026) tested five diversity metrics over 1,140 ensemble
combinations and, after FDR correction, **only double-fault rate predicted ensemble accuracy** —
disagreement rate, Q-statistic, KL divergence and CKA did not. Theisen et al. (NeurIPS 2023) give the
theory: ensemble improvement is linear in the disagreement-error ratio, and DER < 1 predicts no gain.

Our ruler is `MemberDistance`'s rank agreement, which is a disagreement statistic. Its own docstring
already said it "is UNVALIDATED as a predictor and must be read that way". A census over 166 members had
been filed on it two hours earlier.

## The measurement

`fault_ruler` runs Pokhrel's experiment on our own pool rather than adopting his conclusion. Errors are
defined in the space the decision is actually made in: our metric is the mean of twelve ROC AUCs, AUC is
the probability a positive outranks a negative, so a **discordant (positive, negative) study pair** is
the natural unit of "wrong". No threshold, no calibration, no magic number — and it comes with a free
oracle, because a member's mean discordance must reproduce its own `1 - AUC`.

166 members, 48,000 discordance trials each, 1,200 sampled member pairs, rank-average vote.
**Oracle: worst gap 0.0033 against a sampling error of ~0.0023.** The sample is the metric's.

## The first run was a harness finding, not a ruler finding

All four statistics — including the two the literature says point in opposite directions — correlated
**negatively** with measured gain:

```
double-fault rate (negated)              -0.1548
agreement (INCUMBENT)                    -0.5313
disagreement-error ratio (DER)           -0.5610
disagreement rate                        -0.5920
```

Four rulers do not fail at once. One confound does, and this project had already measured it:
`pool_census` shows the distant members of this pool are the **weak** ones. Gain is scored against the
better member, so pairing a strong member with a distant weak one drops the mix below the strong member
alone and books a large negative gain. Every distance statistic was reading "one of these two is bad" —
a fact about the pool's shape, not about diversity.

## The reading, with quality held fixed

Residualising both sides against the pair's quality (the better member's score, and the gap between the
two) in rank space, over the same 1,200 pairs:

```
                                        partial      raw
agreement (INCUMBENT, MemberDistance)   +0.7086   -0.5313
disagreement-error ratio (DER)          +0.6999   -0.5610
disagreement rate                       +0.6955   -0.5920
double-fault rate (negated)             +0.5911   -0.1548
```

Every sign flips. **The incumbent ruler ranks gain best, and double-fault — the one statistic the
citation says is the only one that works — ranks worst of the four.** The ordering here is the reverse
of Pokhrel's.

Measured gain over the pairs: median +0.0031, best +0.0153, worst -0.0986. DER above 1 in 114 of 1,200
pairs, which is Theisen's own prediction of a low-gain regime and is consistent with the +0.0031 median.

## What this licenses, and what it does not

**Keep the ruler.** `MemberDistance` stands, and the 166-member census filed against it stands with it.
The citation was worth testing and does not transfer to this pool.

**But never read a distance number without holding quality fixed.** That is the real finding, and it is
about us rather than about the paper: the raw column above is what a candidate ranking by distance alone
produces, and it is *inverted*. `MemberDistance.report` ranks candidates by distance across the whole
quality range, so its ordering is exposed to exactly this. The census is not — it restricts to strong
members first, which is the control, and that restriction turns out to have been load-bearing rather
than tidy.

**The four statistics are collinear.** 0.5911 to 0.7086 is a narrow spread among measures computed off
the same error matrix, and 1,200 dependent samples do not separate them as cleanly as four decimals
suggest. The safe reading is that the incumbent is *not worse* than the alternatives and that all four
predict gain once quality is controlled — not that agreement is uniquely correct.

**It prices nothing.** These are offline gains on 909 held-out studies, and `record_mix` has already
measured that the holdout prices diversity about 2.4x too cheaply against the board. This ranks
statistics; it does not price a seat and it licenses no slot.

## What it says about the standing plan

Nothing changes on the card. `ortho_stack`'s second bar — become the first occupant of the strong-and-
distant cell that is distant on architecture rather than on decode — was pre-registered against the
agreement ruler, and that ruler has now been checked rather than assumed. The caveat written into
`experiments/arms/ortho_stack.sh` at the same time (read its agreement against the SEATED members, not
against all 166, because its nearest neighbour will be `ortho_dense` itself) is unaffected and still
binds.
