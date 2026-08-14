# The ruler was five times too small, and it pointed at the wrong labels

*2026-08-14. Everything here is CPU-only arithmetic over predictions that were already on disk.*

## What was found

The twenty published CC0 members each carry an `oof` block — the studies they held out and the twelve
probabilities they produced for each. Read together they are **four seeds over five folds**, and the
five id-sets are disjoint with a union of **4,407 of 4,407 studies**: an exact partition. Every study in
the corpus therefore has four held-out predictions from models that never trained on it.

Our ruler until today was one fold: 881 studies, a Hanley–McNeil sampling floor of **0.0053** that has
defined what counts as a result for this whole project. This one is the corpus, floor near **0.0024**.

It answers questions about *combining*, not about whose weights are better — it holds only their
members. That limit is real and does not shrink what follows.

## The label profile is not the one the plan was built on

| finding | fold-0 (881 studies) | corpus (4,407) | 95% interval | positives |
|---|---|---|---|---|
| Lateral Meniscus | not named | **0.7981** | ±0.0159 | 1114 |
| PF OA | not named | 0.8165 | ±0.0121 | 1988 |
| MCL | **0.755 — "target it"** | 0.8166 | ±0.0155 | 758 |
| Lateral OA | not named | 0.8215 | ±0.0138 | 1082 |
| Medial Meniscus | — | 0.8603 | ±0.0102 | 2341 |
| Medial OA | — | 0.8687 | ±0.0105 | 1612 |
| Baker's | 0.896 — "the level to reach" | 0.8808 | ±0.0118 | 1110 |
| **Fracture** | **0.784 — "target it"** | **0.8923 — the best label** | ±0.0187 | 313 |

The ledger's headline target — *"MCL 0.755 and Fracture 0.784 → 0.873 is worth +0.0173"* — is dead.
Fracture is our **strongest** finding. `label_power` predicted exactly this from first principles (an
AUC over 62 positives has a standard error near 0.04, and Fracture read 0.784 / 0.903 / 0.864 across
three folds of one recipe); this is the measurement that confirms it.

**The actual weakest label is Lateral Meniscus, and it was never named as a target.**

## The asymmetry is real, and it is not a bug in our pipeline

Both side-defined pairs favour medial, far outside their intervals:

```
Lateral Meniscus 0.7981 (n=1114)  vs  Medial Meniscus 0.8603 (n=2341)   +0.0622
Lateral OA       0.8215 (n=1082)  vs  Medial OA       0.8687 (n=1612)   +0.0472
```

That is the exact signature a broken laterality normalisation would leave — we mirror left knees onto a
canonical side, and a mis-flipped study swaps medial for lateral. Because medial positives outnumber
lateral roughly 2:1, leakage from the larger class pollutes the smaller one disproportionately, which
would produce this asymmetry and no other.

It survives the test.

```
                       n     med−lat meniscus   med−lat OA
ALL                 4407         +0.0621          +0.0471
source = tag        2203         +0.0475          +0.0369
source = geometry   2132         +0.0851          +0.0586
tag + geom AGREE    2105         +0.0519          +0.0331
```

The gap persists at +0.052 / +0.033 among the 2,105 studies where the DICOM `Laterality` tag and the
independent geometric rule **agree with each other**. Those are as confidently sided as a study can be;
a flip bug cannot hide inside them. The dead zone where neither rule resolves is 72 studies (1.6%) and
the two rules disagree on 30 — together far too few to move an AUC by 0.06.

The geometry-decided subgroup shows a larger gap, and it is tempting. It is also **unmatched**: its
*unsided* labels score 0.8748 against the tag subgroup's 0.8565, so those studies are simply easier
overall. Directional at most, and not something to plan around.

So the innocent rival wins: lateral-compartment findings are genuinely harder or rarer, and the
normalisation is doing its job. What this buys is a **targeting** statement rather than a fix — and one
that clears the bar. Lifting the four weakest labels to 0.87, a level two other findings already reach,
is worth **+0.0189** on the mean.

## Our three label tables are one ruler

`confident_v2`, `flatyes_v2` and `report_labels_v2` differ in **graded** values on 45% and 27% of cells
— and their **binarised** targets are identical everywhere: the same 16,058 positives, and mean AUC
equal to four decimals. They differ only in the supervision they give training, and can never move an
evaluation number. Any experiment framed as "compare label tables" is comparing training signal alone.

## What this cost to not have

The oracle that reads these blocks existed once as `scratchpad/oof_oracle.py`, was never committed, and
is gone along with the weights it ran on. Two beads argued for three days about a residual from a number
nobody could recompute, and both were settled today by facts sitting inside checkpoints we had never
opened — the published members declare `crop_mm 130.0`, `band 0.20,0.80`, the identical slot order and
byte-for-byte the same `rules` dict we hardcode.

The order of operations was also backwards twice. The contested-slot census cost 4,407 DICOM header
reads and answered whether a routing divergence was *possible*; comparing the two orderings cost one CSV
read and answered whether it *happens* (it does not — the table order and our directory order are the
same for all 4,407 studies). The cheap measurement dominated the expensive one and was available first
both times.

## What is now true, and what is still open

- **Settled**: the slot tie-break cannot explain the residual; no declared reading axis differs; the
  MCL+Fracture target is dead; laterality normalisation is not the cause of the medial/lateral gap.
- **Open**: the 0.011–0.025 residual against identical weights, which must live in the *implementation*
  of the reading rules rather than any recorded policy — the rebuilt `oof_oracle` is pointed at it and
  needs one GPU pass.
- **Open**: whether per-label combining helps *diverse* members. The estimator is calibrated — over four
  seed twins it correctly declines to reward selection (split-half 0.8403 / 0.8450 / 0.8468 / 0.8475,
  losing monotonically toward the flat average) — but seed twins are the null case, and the real test
  needs our own members' predictions cached once.
