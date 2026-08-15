# The paired floor is 2.2× tighter, not eleven — and that was our own estimate

*2026-08-15. CPU arithmetic over the published out-of-fold predictions. No GPU, no training, no slot.*

knee-3vh has argued since 2026-08-13 that this project cannot read its own per-label results. Every
per-label delta was quoted bare; calibrated against each label's unpaired error bar, four arms with
unrelated mechanisms each "improved" Fracture inside a ±0.061 band. The bead's proposed fix was the
paired interval — two arms scored on the SAME studies share their sampling error, so the difference is
far better determined than either AUC — and it estimated the gain at **"about eleven times tighter"**.

Nobody had measured that. It is 2.2×.

## The measurement

The published members are four seeds over five folds whose `oof` id-sets partition all 4,407 studies, so
every study carries predictions from models that did not train on it. Two seeds of one recipe are a
**null pair**: the true per-label difference is zero, so the bootstrap interval across them is the
instrument's own width. That is what a floor is.

| | paired width | unpaired width | tighter |
|---|---|---|---|
| **corpus, 4,407 studies** | 0.0141 | 0.0308 | **2.2×** |
| **our ruler, 881 studies** | 0.0323 | 0.0683 | **2.2×** |

Per label at 881 studies, the paired width runs from 0.0176 (Medial OA) to 0.0501 (Fracture) — half-widths
of ±0.009 to ±0.025, mean **±0.016**.

**Sanity check on the instrument, since the result contradicts the estimate it replaces.** Both widths
should scale as 1/√n between the two holdout sizes. Unpaired: 0.0683/0.0308 = 2.22. Paired:
0.0323/0.0141 = 2.29. Against √5 = 2.236. The bootstrap is measuring sampling error and nothing else.

## Why 2.2× and not 11×

The estimate assumed two arms' errors cancel almost entirely. They do not, because a *label's* AUC error
is dominated by WHICH studies are in the resample — how many Fracture positives, and how hard they are —
and both arms suffer that together only to the extent their rankings agree. Correlated, not identical.
Pairing removes rather less than half the width, twice over, and stops.

## The part that makes this a floor and not a verdict

**Seed twins are the most correlated pair that exists here.** Two arms with genuinely different
mechanisms — a different crop, a different resolution — rank studies less alike, so their paired
interval is *wider* than this one. So 2.2× is an **upper bound on the tightening** for any real arm
comparison, and the eleven-fold claim is refuted a fortiori: no real pair does better than this.

## What it changes for the ledger

At a mean paired half-width of ±0.016 on our 881-study ruler, applied to the two per-label predictions
knee-3vh says it inherits:

- **knee-8ev, extended slots**: predicted the gain concentrates in the coronal-read labels; measured
  MCL +0.0056 against a paired half-width of ±0.0236. Still **unread**. Pairing does not rescue it.
- **knee-bot, calibrated targets**: measured PF OA +0.0198 against a paired half-width of ±0.0136.
  That **clears**. The arm's falsifiable claim still fails — it predicted MCL and Effusion, and PF OA
  was anti-correlated with the prediction — but the PF OA movement itself is not noise.

**Both readings are indicative, not settled, and the reason is worth stating plainly:** this floor was
measured on published seed twins, and applying it to our arms assumes our arm pairs are as correlated as
twins are. They are not. Settling either needs each arm's own paired interval, which needs its per-study
predictions.

## Which turns out to be the real blocker, and it is not statistics

Auditing knee-3vh against the repo rather than from memory: **the measurement it asks for already
exists.** `LabelScores.paired_interval` (`kneemri/scores.py:86`) has been there, seeded and unit-tested,
and `member_curve` already consumes it. `PredictionCache` exists for exactly this, and its own docstring
names the problem — *"the ANALYSIS half of the project was gated on a GPU it does not need"*.

What is missing is the data. `processed/holdout_predictions/` **does not exist on disk**; the cache has
never been written. And `run_comparison` still prints the unpaired bound for a structural reason rather
than an oversight: it reads ledger rows, which carry per-label AUCs and no predictions, and a bootstrap
cannot run on a summary statistic.

So the honest split of that bead's acceptance is: the statistics half is **done**, and the "re-read the
arms already run" half needs one GPU inference pass to populate a cache — after which every arm on disk,
and every arm run afterwards, is readable at paired precision for free.

## The pattern, again

An estimate written into a bead as a parenthetical — *"about eleven times tighter"* — became the reason
to prioritise the work, and stood for two days. Measuring it cost one CPU run against files already on
disk. It was wrong by a factor of five, in the direction that matters: the instrument is much less
powerful than the plan assumed, so several deltas this project hoped pairing would rescue stay unread.
