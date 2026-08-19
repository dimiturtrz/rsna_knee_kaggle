# Pooling is closed on the mean, but not on Lateral Meniscus

Two arms landed on 2026-08-19/20 asking the same question from opposite ends. `attend_dense` replaced
the pooling OPERATOR — the elementwise max over slices — with a learned 64-wide gate. `focal_dense`
left the operator alone and widened the pooled vector, concatenating a focal tail. Both are single-
variable pairs against `band_dense`, verified in the checkpoint config and not only in the recorded
recipe. Both closed their cell of the decode family: −0.0009 and −0.0044 on holdout mean AUC, and
together they retire re-allocation as an explanation of the substitution shape (`experiments/README.md`).

This note is about the thing that did not close.

## A second ruler that costs nothing

`kneemri.evaluation.ordinal_signal` asks a different question of the same arrays already on disk: among
the studies a member never trained on, how often does it rank a 0.94-graded study above a 0.68-graded
one? It needs no card and no labels file, and it is independent of holdout mean AUC — one is a
present-vs-absent ruler, the other a severity-ordering ruler over a band the binary ruler collapses.

Run over all three members it says the same thing the binary ruler does about the MEAN. Against
`band_dense`, `focal_dense` is **−0.0047** in mean ordinal concordance (matching its −0.0044 binary) and
`attend_dense` is **−0.0002** (matching its −0.0009). So there is no hidden axis here: neither arm bought
severity ordering that the binary ruler was too coarse to see. That was the live possibility worth
checking, and it is answered no.

## The one label that moves on both rulers in both arms

| label | binary Δ, `focal_dense` | binary Δ, `attend_dense` | ordinal Δ, `focal_dense` | ordinal Δ, `attend_dense` |
|---|---|---|---|---|
| **Lateral Meniscus** | **+0.010** | **+0.009** | **+0.017** | **+0.018** |
| ACL | +0.006 | +0.005 | −0.001 | −0.019 |
| Lateral OA | +0.005 | +0.007 | −0.009 | −0.004 |
| Medial Meniscus | −0.006 | −0.004 | +0.006 | +0.002 |
| Baker's | −0.003 | −0.009 | +0.007 | +0.006 |
| Fracture | −0.013 | +0.008 | −0.035 | +0.023 |

Lateral Meniscus is the **only one of the twelve labels positive on all four readings**, and the four
numbers are tightly clustered (+0.009 to +0.018) rather than one large value carrying three small ones.
Base rates: `band_dense` holds Lateral Meniscus at 0.846 binary and 0.697 ordinal, its second-lowest
ordinal score of twelve.

## Why this is not the best-of-twelve artefact it looks like

It would be, on its own. Under a null where each label is a coin flip on each ruler, the chance that a
GIVEN label comes back 4/4 positive is 1/16, so among twelve labels the expected count of 4/4 labels is
about 0.75 — finding exactly one is what chance produces, and reading it as a discovery would be picking
the winner after the race.

What removes that objection is that **Lateral Meniscus was pre-registered.** `focal_dense.sh`, committed
before the run, names it in its prediction: *"The named riser is Lateral Meniscus, the largest model-side
gap in the corpus (−0.1932, knee-dtk) and a 1–3 mm finding, with Fracture second."* The label was chosen
from a mechanism — a focal tail exists to rescue a fine-scale response a mean over 576 tokens divides
away, and Lateral Meniscus is the finest-scale finding with the largest gap — not from the results table.
The other eleven rows above are post-hoc and carry the multiple-comparison penalty in full; this one does
not. The pre-registration's SECOND named label, Fracture, went the other way in `focal_dense`
(−0.013 binary, −0.035 ordinal), so the prediction was not vague enough to be unfalsifiable.

It also replicates across an arm that never named it. `attend_dense` was written about the pooling
operator and says nothing about Lateral Meniscus; it moves the label by +0.009 and +0.018 anyway. Two
mechanisms — a more expressive allocator, and a wider vector — both touch the same label, which is what
a real effect looks like and what a lucky one usually does not.

## What it does and does not license

It does **not** license another pooling arm. Both arms lost on the mean, both seat at +0.0015 against the
shipped four (an ADD, not a seat, under `comp/champion.json`'s four-member principle), and tonight's
board read says an offline edge inside the 0.0062 floor is not worth a submission slot. Nothing here is
shippable.

It does **not** license tuning toward Lateral Meniscus. Fitting the objective to a named weakness is the
thing this project refuses to do; the fix for a weak label is balanced data, not loss-engineering.

What it records is **where the pooling family acts.** The family is closed as a MEAN lever and the
ledger says so. It is not closed as a statement about fine-scale findings: changing how slices are
pooled moves the finest-scale label and pays for it everywhere else, on two independent rulers. That is
the third independent line pointing at Lateral Meniscus, after knee-dtk's gap measurement and the two
`lateral_gap_is_not_*` notes that ruled out prevalence and the crop as causes. Whatever is limiting that
label is upstream of pooling — pooling can shift budget toward it but cannot create it — which is the
same direction the two closed arms send the bottleneck: supervision, or the encoder.

## Limits

- One holdout split, 909 studies, one seed per arm. Per-label AUC noise is wider than the 0.0062 mean
  floor; no per-label interval is quoted here because none was computed, and the argument rests on
  pre-registration plus replication across arms rather than on any single delta clearing a bar.
- The ordinal ruler's denominators are cross-level PAIRS among graded studies, 15,296 for Lateral
  Meniscus — not the holdout's 909 studies. Those pairs come from far fewer studies than the count
  suggests, so the two rulers are not independent in their sampling, only in what they measure.
- Both arms are compared at their own best epoch (8 and 9 against the control's 8), which is the
  comparison the ledger uses throughout and which slightly favours whichever arm has the noisier curve.
