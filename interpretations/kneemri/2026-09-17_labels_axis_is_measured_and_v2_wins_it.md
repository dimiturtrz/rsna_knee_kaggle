# The labels axis is measured eight ways, and the shipping default wins it

2026-09-17. Written because "labels are the lever" has been raised repeatedly and abandoned repeatedly,
and the owner asked for the reason rather than another restatement. The reason is not loss of nerve.

## The measurement

`experiments/lib.sh:72` sets `LABELS="$PROCESSED/labels/report_labels_v2.csv"` GLOBALLY, so every arm in
the repo inherits it unless it overrides. That is why `labels` never shows up as a differing leaf against
the shipping corner, and why the key looks unexamined: it is the one leaf that cannot differ by default.

The `*_span` family is the axis measured at a single corner. All eight arms carry a byte-identical flag
line — `$RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT --backbone-lr 4e-5 --slices 12
--band-low 0.1 --band-high 0.9` — differing only in `--labels`, and every challenger carries
`--eval-labels report_labels_v2.csv` so all eight are scored against one truth. Single-knob, one ruler.

`ledger75 --holdouts D:/models/rsna_knee_abnormality`, population 66c6f231, 909 studies:

```
band_span_s2026        report_labels_v2.csv (the default)   0.8851   peak e6
confident_span_s2026   confident.csv                        0.8785   peak e9   -0.0066
v8named_span_s2026     report_labels_v8named.csv            0.8766   peak e8   -0.0085
v2delonly_span_s2026   report_labels_v2_delonly.csv         0.8749   peak e6   -0.0102
discord_mask_s2026     discordant_masked.csv                0.8629   peak e6   -0.0222
v7nontear_span_s2026   report_labels_v7nontear.csv          0.8612   peak e6   -0.0239
v5blend_span_s2026     report_labels_v5blend.csv            0.8602   peak e5   -0.0249
v6tears_span_s2026     report_labels_v6tears.csv            0.8580   peak e6   -0.0271
```

The shipping default is FIRST, by +0.0066 over the runner-up — just over the 0.0062 paired floor.

## What this closes, and what it does not

`recipe_pairs --key labels` reports 30 pairs, 17 clearing the floor, which is by far the largest block in
the 31-key census and reads like the biggest unexploited axis in the repo. It is not. The printer caps at
eight pairs per key and sorts by magnitude, so every pair it shows is a comparison BETWEEN LOSING VARIANTS
— v6tears against v5blend against v7nontear, deltas of 0.015 to 0.027. Those are differences among bad
label sets. None of them is headroom above v2, and v2's own pairs sort below the cut precisely BECAUSE
they are small. A large pair count on this key measures how far the failed attempts fell, not how far
the leaf could rise.

So SELECTING among the label sets this project owns is finished. Seven alternatives were built and every
one of them lost. Re-opening the axis means BUILDING a label set that is not in this list, and it must be
argued on a mechanism, not on the size of the `labels` pair block.

## The confound that keeps this axis alive, and why it is not resolvable from this table

Every challenger is scored with `--eval-labels report_labels_v2.csv`. That is the correct design for a
single-knob comparison — one truth for all eight — but it has a consequence that cannot be measured from
inside it: a label set genuinely BETTER than v2 would score WORSE here by construction, because the ruler
it is being measured against is the thing it improves on. The table establishes that nothing here beats
v2 AT PREDICTING V2. It cannot establish that nothing here beats v2 at predicting the competition truth.

That is the honest reason this lever keeps being raised and keeps dying: the result is real and the
question it answers is narrower than the question being asked. Closing it for good needs a ruler that is
not v2.

## The board IS that ruler, and it resolves this — an arithmetic correction

This section first said the board "costs a slot per reading at 0.001 resolution, which is coarser than
the 0.0066 being resolved." That compares an OFFLINE delta against a BOARD resolution without passing it
through the transfer, which is the same class of error `CLAUDE.md` records against the compression
constant four times. Corrected: on the ledger75 fit the whole-record slope is **+1.10 with residual
sd 0.0017**, so 0.0066 offline75 lands as **+0.0073 board — seven ticks**. Two readings differ by
√2 × 0.0017 = **0.0024**, so the predicted gap is **three sigma**. The board resolves this axis.

And it is the RIGHT ruler for it, not merely an available one. The confound above is that every
challenger is scored against the thing it claims to improve on; the competition's truth was written by
neither v2 nor any challenger, so a label set that is genuinely cleaner has somewhere to show it. The
prediction splits the hypotheses cleanly: if v2 is simply better labels, `band_span` beats
`confident_span` by about seven ticks, as the offline table says. If `confident.csv` is cleaner and only
LOOKED worse because v2 was the ruler, the order INVERTS — and an inversion cannot be produced by the
scatter, which is a third of the gap.

Cost is two slots and NO card: both checkpoints have sat on disk since the family was run
(`band_span_s2026.pt`, `confident_span_s2026.pt`), byte-identical flag lines differing in `--labels`
alone. That is the cheapest decisive reading available on any open axis.

## Consequence

SELECTION among the owned label sets is finished at the shipping corner, and composition work should
look elsewhere — `[[head_lr-at-the-shipping-corner]]` (knee-xuarh) is where that search landed. But the
axis is NOT closed, and the reason it is not is the confound, not a loss of nerve. The reading that
closes it is priced above and owes no card.
