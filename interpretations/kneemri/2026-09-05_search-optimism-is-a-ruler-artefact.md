# The search was never the problem: selection optimism at 909 studies is +0.0009

**2026-09-05, 22:56Z. Card-free, three minutes of CPU.**

## What was believed this morning

That a greedy blend search over the 195 priceable members, scored on the 909-study shared holdout, was
optimistic by about **+0.0201** — and that every searched offline number in `comp/board.jsonl` therefore
needed a correction of that size before it could be compared to anything.

That figure came from `priceable_blend --optimism`, which cut the 909 studies in half by sorted position,
selected greedily on 455, and priced the winners on the 454 it had never seen: selection half 0.8841,
held-out half 0.8641.

It sat awkwardly beside `kneemri/evaluation/selection_bias.py`, which reads **+0.0025 to +0.0031** from
four `MemberCurve.honest` replications at 7-12 seats over pools of 77-119 members, and which had been
cross-checked on the board itself: `greedyseat_v1`'s raw offline predicted its board score +0.0027 too
high, and the bias-corrected offline missed by +0.0001. An offline measurement of +0.0028 and a board
measurement of +0.0027, on two rulers that share nothing.

An eightfold disagreement between two live measurements is not something to average. The reading recorded
at the time was that both were real and they measured different **regimes** — the split-half had moved
two knobs at once, halving the studies AND using the widest pool — so +0.0201 was an **upper bracket**
and the honest statement about the target was a bracket too: the single-model step to 0.953 was somewhere
between +0.013 and +0.027.

## The measurement that closed it

Hold the pool at 195. Give the search back its studies. `priceable_blend --optimism --folds 10`:

| | selected on | priced on | in-sample | held out | gap |
|---|---|---|---|---|---|
| fold 1 | 818 | 91 | 0.8747 | 0.8886 | −0.0139 |
| fold 2 | 818 | 91 | 0.8781 | 0.8631 | +0.0150 |
| fold 3 | 818 | 91 | 0.8782 | 0.8530 | +0.0252 |
| fold 4 | 818 | 91 | 0.8764 | 0.8786 | −0.0022 |
| fold 5 | 818 | 91 | 0.8740 | 0.8951 | −0.0211 |
| fold 6 | 818 | 91 | 0.8761 | 0.8733 | +0.0028 |
| fold 7 | 818 | 91 | 0.8758 | 0.8872 | −0.0114 |
| fold 8 | 818 | 91 | 0.8770 | 0.8697 | +0.0073 |
| fold 9 | 818 | 91 | 0.8755 | 0.8862 | −0.0107 |
| fold 10 | 819 | 90 | 0.8778 | 0.8596 | +0.0182 |

**Mean +0.0009. Spread −0.0211 to +0.0252. Standard error ≈ 0.0046.**

+0.0201 is four standard errors outside that interval. It is **refuted** as an estimate of what a search
run at full data overfits. It survives only as a description of a different quantity: what a search
overfits when the ruler it reads has been halved — a regime no number in `comp/board.jsonl` was taken in.

`selection_bias.py`'s +0.0025 to +0.0031 sits inside the interval. Two estimators that share no code
agree, and the board cross-check agrees with both.

## The mechanism, which is the part worth keeping

Read the two score columns rather than the gap column:

- in-sample: 0.8740 to 0.8782, **spread 0.0042**
- held out: 0.8530 to 0.8951, **spread 0.0421**

The in-sample side barely moves — ten independent greedy searches over 818 studies each land within
0.004 of one another. Ten times as much of the per-fold variation is on the **held-out** side, where 91
studies is simply too few to score a blend. So the fold-to-fold gap is dominated by the ruler moving, not
by the search finding anything, and the +0.0201 from a single split was one draw from a distribution that
is wide for a reason unrelated to selection.

The second half of the mechanism is in the seat lists. Four members — `stack_increments`,
`labelwise_dense`, `margin_seat`, `blendhold` — are selected in **all ten** folds. `finding_calib_dense`
appears in seven, `warm_calib_lowlr` in five. The search is stable under resampling of 90% of the corpus,
and a stable search is a search with little left to overfit. That is a statement about this pool: 195
members deep, with a handful that are genuinely better than the rest and are found every time.

## What it changes

**The target.** The single-model step from the pool's best member (0.8642) to a 0.953 board score is
**+0.015 of honest offline** — not a bracket, and not the +0.027 that went to the peer at 21:28Z. Both
earlier figures are corrected in `ceiling5_v1`'s row and in a third letter.

**`ceiling5_v1` itself needs no revision.** Its row already recorded 0.8729 (raw 0.8757 minus the
selection_bias band) on the grounds that the band was the defined correction; the K-fold says that was
the right call for the right reason, which is a better outcome than a number that happened to land well.

**A default changed in the tool.** `priceable_blend --optimism` now defaults to ten folds, and its
docstring states why: the count is the measurement's second variable, and a two-fold reading is an upper
bracket on the quantity the module is asked for, never its value.

## What it does not change

Nothing about the board. `ceiling5` still predicted about 0.934 through the fold-0 anchors and read
0.934; the transfer ruler is still not inverted; the entry still refutes nothing else.

And nothing about how wide the bands on a **single** fold's reading should be. A 91-study ruler swings
0.042 across resamples of the same models. Any future number priced on a subset that size is a direction,
not a value — which is the same warning the project already carries about the 58-study gold ruler, now
measured on the proxy as well.

## The methodological point, stated plainly

The correction was wrong because the measurement of it moved two variables at once, and it was believed
for a morning because the number it produced was large enough to be interesting. The fix cost three
minutes of CPU and required no card, no slot and no new data — only holding one knob still. Reading the
per-fold table rather than the summary line is what showed the reason, and the reason (the ruler, not the
search) is more useful than the corrected number.
