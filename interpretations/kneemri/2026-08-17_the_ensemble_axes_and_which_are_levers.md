# The ensemble has six design axes. Five are now measured, and only one is a lever.

2026-08-17. Ensembling is the largest single effect this project has found — one member scored 0.868 on
the board and five weaker ones scored 0.888 — so "how many members" gets asked a lot and the other five
questions do not. This is all six, with what each is worth.

The bar throughout: a result needs ~1.5% absolute. The board's own noise floor is 0.0053 and the
fine-tuned three-seed floor is 0.0062. Board points compress against holdout points at a **median 4.67:1**
(knee-7d1), so a holdout delta is multiplied by ~4.7 to read as a board delta.

All offline numbers below are fold 0 of `report_labels_v2`, six all-ours members, cached probabilities.

## 1. Count — measured, saturates at four

```
1 member   0.8438  [0.8215, 0.8523]
2 members  0.8529  +0.0091   <- the largest increment on the curve
3 members  0.8562  +0.0034
4 members  0.8580  +0.0017
5 members  0.8590  +0.0010
6 members  0.8598  +0.0008
```

The 1→2 step is bigger than every method change this project has landed except the augmentation fix
(+0.0110). Past four, each member buys less than a fifth of the noise floor for a full inference budget.

This is also the answer to "what if we train just one?" — measured directly on the board, where a solo
member scored **0.868** against five weaker ones at **0.888**. Offline, dropping from the 0.919 pair to
the best single costs 0.0058, which is ~0.027 on the board: roughly 0.892, or rank ~710 of 1,758 instead
of 84.

## 2. WHICH members — measured, small and free

Marginal selection (sort by solo AUC, take the top k) against joint selection (greedy on ensemble AUC).
Four members: `strongest(4)` = 0.8588, joint-chosen = 0.8612. The sort drops `intensity392_ws` (solo
0.8459) in favour of `extended_ws` (solo 0.8462) and loses **0.0024** doing it — the marginally better
member is the jointly worse one, which is the whole argument for joint selection in one line.

Free (it is a sort over arrays already in memory), directional in size, and worth taking. The board could
not resolve it: joint-4 scored 0.918 against the pair's 0.919, a tie.

## 3. Diversity axis — measured, and the only lever

WHAT makes members differ: decode path / pixel group, architecture, target transform, window.

Collapsing three members onto one decode path cost **0.0059 offline and 0.014 on the board — 2.4x the
usual compression**. The holdout systematically underprices this axis, which means every offline
measurement of diversity is a lower bound on what it is worth.

It is also the axis that closed the gap to the frontier. Five all-ours members reached 0.911 and two
reached 0.919, both above the mixed 22-member fusion's 0.895 by more than the bar — so the gain belongs
to the member pool, and the CC0 weights we started from are now a cost rather than a contribution
(knee-k5kk).

Diversity is not the same as *foreign*. It is the same as *failing differently*.

**But failing differently is necessary, not sufficient — measured the same day.** The one member we own
that reads a different pixel group, `crop80_tissue` (80 mm about the tissue centre, 0.238 mm/px), was
built to fail differently and does. Added to the six as a seventh member it costs **−0.0003** (0.8598 →
0.8595), and joint selection never wants it: best-4 of the six is 0.8612, best-6 *including* crop80 is
0.8607, best-7 is 0.8595. Every subset containing it scores below the best subset of the same size
without it. Its solo gate had already refuted the mechanism on 2026-08-16 — the diffuse half of the
prediction landed and the small-structure half did not — and the marginal read reaches the same verdict
by an independent route.

That does not kill the axis. crop80 ran eight epochs against the pool's twelve (a mismatch priced
elsewhere at −0.0028), so this measures an undertrained member of the crop family rather than the crop
family; `centre_tissue` has never run; and the slice BAND has never been varied by anything. What it does
kill is "a crop-diverse member is a free bet". The next one has to be schedule-matched before its number
means anything.

## 4. Combiner form — measured 2026-08-17, FLAT

The axis that had never been varied: every number this project produced came out of one rule, a flat
per-label rank-average, chosen because the metric reads rank order. That argues for ranking. It does not
argue for giving six members an equal vote when 0.031 separates the best from the worst.

```
                        six members   the 0.919 pair
RankAverage (shipped)      0.8598         0.8581
LogitAverage               0.8603         0.8583
QualityWeightedRank        0.8600         0.8580     (split-half, weights fitted off the scored half)
```

Every spread is an order below the board floor. Keeping the confidence margin a rank discards buys
+0.0005; weighting by what a member actually knows buys +0.0002. **The combiner is not a lever** — the
flat rule is leaving nothing on the table, and the gains are in the members rather than in how their
votes are counted.

Not ruled out: a stacked combiner fitted across folds, and a combiner over members that disagree far more
than these six (which share a recipe lineage). Neither is worth building before the pool diversifies.

## 5. Per-label member selection — measured, DEAD

Choosing a different subset for each of the twelve findings. Its **oracle sits 0.0008 BELOW** the plain
rank-average (0.8590 vs 0.8598) — the ceiling is beneath the floor, so there is nothing under it to
collect even with the answers in hand. Split-half confirms: top-4 per label reaches 0.8615 in-sample and
0.8615 honest, against the flat rule's 0.8598, and the in-sample column shows the optimism is the gain.

Twelve choices over a few hundred studies, some labels carrying 62 positives, is not enough to fit
anything. Recorded as knee-vjmw; do not re-open without more holdout.

## 6. Passes per study / TTA — measured, depends on the member

Overlap TTA is worth **+0.0146 to a windowed member and 0.0000 to a whole-stack one**. A whole-stack
member already reads the volume in one window, so window averaging is structurally unavailable to it and
rotation is the only test-time average it can have. Since the current pool is whole-stack, this axis is
currently worth nothing — which is a fact about the pool, not about TTA.

## What this leaves

One lever (diversity of the decode path, underpriced offline by 2.4x), one free tweak (joint selection,
+0.0024), and four axes measured out. The next real gain is a member that fails differently from the six
we have — not a better vote, not a per-label rule, and not a seventh member of the same kind.

And the lever is narrower than "read different pixels". The one differently-reading member we own
subtracts, so what the axis needs is a member that fails differently AND scores in the pool's range —
which crop80, at 0.8345 against a pool floor of 0.8459, does not. Diversity buys nothing from a member
the ensemble has to carry.
