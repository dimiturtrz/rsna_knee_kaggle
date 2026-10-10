# Post-processing is spent on the holdout; per-finding share is the one lever left (2026-10-10)

Ruler: the 896 studies shared by our six seated members' holdout records and two honest replicas of
nartaa's recipe (`raptor_ft_repl2x2_{a,c}_swa`), ledger75 cut (midpoint on Medial Meniscus and Baker's).
The replicas read 0.8787 here while the real checkpoint beats our seven on the board (0.949 vs 0.945),
so every number below that involves the foreign side is an ORDERING, never a level.

## What was tried, and what it read

| Method | Fit | Δ macro vs its base |
|---|---|---|
| cross-label stack, ours 12 logits → each label (logistic, C 0.01/0.1/1) | 5×5-fold CV | −0.019 / −0.007 / −0.008 |
| per-label member reweight (6 members) | 5×5-fold CV | −0.002 / −0.001 / −0.003 |
| members + cross-label | 5×5-fold CV | −0.003 / −0.004 / −0.009 |
| seat, cross-label stack over 24 | 5×5-fold CV | −0.011 / −0.004 / −0.009 |
| truth-correlation coupling (score + α·corr·others), α 0.05–0.4 | 5-fold CV | −0.0003 … −0.011, monotone in α |
| aggregation rule over our six: rank / median / max / min / power-mean | none | −0.0003 / 0.0000 / −0.010 / −0.007 / −0.002…−0.007 |
| max(ours, foreign) rank vs ½ rank mean | none | −0.0007 [−0.0036, +0.0023] |
| per-label share of the pair (logistic on two z-scores) | 5×5-fold CV | **+0.006** |

The models already carry the label correlations (truth corr up to 0.38: Effusion–Synovitis,
OA trio, Medial Meniscus–Medial OA) — re-imposing them only adds the fold's noise. The one positive is
per-finding SHARE, which the board is already being asked about (`mirrortilt`, `mirrortiltthird`), and
its size here is a replica's, not the real checkpoint's.

## `mirrorlogit` is a share knob, not a confidence lever

Its docstring reads logit-at-½ over rank-at-½ as +0.004 "cross-head confidence". On the same 896 studies
our averaged logits are 1.3-2.2× wider than a replica's per label (1.87 vs 1.21 overall), so a logit mean
hands ours roughly ⅔ of each study's vote. Rank at ours ⅔ reads **0.9078 / 0.9076** (replicas a / c);
logit at ½ reads **0.9076 / 0.9075**. Scale-free z-logit at ½ gains only +0.0009 over rank. The whole
"gain" is the ruler's monotone preference for ours share, which this ruler has because it understates
nartaa, and which the board does not share (¼, ⅓, ½ read 0.952 / 0.953 / 0.953). Not submitted; voided
as a candidate. Its private risk (0.24) is also the ⅔ row's (0.25), as the identity predicts.

## Errors

- **Fracture** is our weakest label (0.842; 39 positives) and the only one where even the weaker replica
  beats us (0.870). Every combination that moves weight foreign helps it on the holdout, CIs wide
  (rank ¼-ours +0.028 [0.000, +0.061]). Two public notebooks special-case Fracture the same way
  (`mattiaangeli/bend-the-knee-to-dinov3-ensembled` keeps it from the base family; `prvsiyan/…read-the-report…`
  a "Fracture max" rule). `mirrortiltthird` gives ours ¼ there.
- **Medial Meniscus** false negatives (29 of 475 positives ranked in our bottom 30%) are 24 at soft truth 0.68
  and 6 at 0.82 — the grade just over the cut. Label ambiguity, not a ranking a post-process can fix.

## Private risk by candidate (knee-kl48k)

`P(private below the pile | public lead +0.003)`, 300 bootstraps of 910, both replicas agree to ±0.01:
ours ¼ 0.04 · ⅓ 0.09 · tilt@⅓ 0.14 · ½ 0.18 · tilt@½ 0.21 · logit ½ 0.24 · ⅔ 0.25 · ours alone 0.31.
Spread of the lead across 1.5T/3T/Siemens/Philips/GE is ≤ 0.0045 for every blend — no stratum flips one.
So lower ours share buys private safety monotonically; a per-finding tilt costs ~+0.05 risk over its
centre share.
