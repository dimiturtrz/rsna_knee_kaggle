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

## The public wave holds no new seat (read 06:35Z)

`rabari9999/rsna-knee-sota-0-954`: 0.954 is the notebook's stated TARGET, no score shown; members are
nartaa-384, nartaa-224 (0.30 inside the CoAtNet side) and goodpjw2008's reader — `triseat`'s, which read
0.952 — and the only change from its own `lb-0-95` is the reader share on 8 findings (Lateral OA 0.28→0.50)
and dropping the probit half of the fusion. A per-finding share retune found on the public board.
`heliosli/*` (and `evgendvorkin/rsna-versia-7`, verbatim heliosli): 0.7 nartaa + 0.3 an OrthoFoundation-L /
DINOv3 compact model; its weights sit in `heliosli/rsna-knee-ofl-ssl-weights` (403 to us) and a
`rsna-knee-compact-v1` whose public copy is one README, licence `other`, titled "Private Weights". Not
mountable, not licence-clean.

Re-read 07:28Z, four newer notebooks, still nothing: `jenillangaliya/…dual-stage…lb-0-951` and
`rainycloudss/…apex-grandmaster-stack-v9` are nartaa-384 0.70 + goodpjw2008's ConvNeXt reader 0.30, i.e.
the wave `triseat` already seats; `omn1v3r53/omniverse-sota…` mounts nartaa + `tonylica/…bend-dinov3-0917-repro-assets`
(mattiaangeli's DINOv3 family, closed by `bend_v1`) but its code runs a single-slice ResNet stub with a
hash-of-UID fallback, and its 0.9572 ablation table has no run behind it; `innshrestha/…0-943` is the
older raptor/DINOv2 pile at 0.943.

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

## A mirror-family second pick hedges nothing

Correlation of the private-minus-pile delta with `mirrorthird`'s, same 400×910 bootstraps, both replicas
agreeing to ±0.01: ours ¼ **0.99**, ½ **0.99**, tilt@⅓ **0.95**, ours alone **0.94**. Two picks that fail on
the same draws are one pick. The levels here favour ours (the replicas understate nartaa), so the expected
max of a pair is not readable — only the correlation is. Pick 2 is therefore worth a different seat
COMPOSITION if one reaches 0.953 on the board; failing that it is `mirrorseat` (½) as the upside pick, and
the hedge it buys is small.
