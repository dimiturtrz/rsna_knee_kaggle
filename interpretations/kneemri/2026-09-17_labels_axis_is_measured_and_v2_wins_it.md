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
not v2 — held-out annotations, or the board itself — and the board costs a slot per reading at 0.001
resolution, which is coarser than the 0.0066 being resolved.

## Consequence

The labels leaf is BANKED at the shipping corner, not unexploited. Composition work should look
elsewhere; `[[head_lr-at-the-shipping-corner]]` (knee-xuarh) is where that search landed instead.
