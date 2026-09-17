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

## The non-v2 ruler already existed, and it does not invert the order

The section this replaces priced a two-slot board reading as "the cheapest decisive reading available on
any open axis". That was wrong twice over. `confident_span_s2026.pt` is on neither shipping shelf
(`MemberPackage.shipped` resolves exactly two datasets), so the arm could not have been built as priced.
And the reading itself was already on disk: `research/scripts/rescore_t1_ruler.py` scores the whole pool
against a CORRECTED truth for zero slots and zero card.

The corrected truth comes from `v2_lone_dissent.csv`, the cells where the report-cluster and v2 disagree
and v2 stands alone. Three rulers, differing only in WHICH of those cells are applied to the 909:

- **T0** — v2 itself, the ruler the family was trained and scored against. 0 cells.
- **T1** — all 469 reference-matching cells flipped. 424 deletions plus 45 additions.
- **T2** — the 424 deletions only.

T2 exists because the two halves are not one operation. D363 adjudicated 56 blind cells STRATIFIED BY
DIRECTION before any verdict existed: deletions (v2=1 -> cluster 0) 2/28 wrong; additions
(v2=0 -> cluster 1) **22/28 wrong**, 21 of those 22 OA columns on one mechanism. So T1 applies one half
that fixes and one half that breaks, and T2 is the ruler that adjudication licenses.

```
                        T0       T1       T2      T2 - T0
band_span_s2026       0.8565   0.8592   0.8579   +0.0014
v2delonly_span_s2026  0.8417   0.8555   0.8545   +0.0128
v8named_span_s2026    0.8468   0.8509   0.8501   +0.0033
confident_span_s2026  0.8449   0.8480   0.8471   +0.0022
discord_mask_s2026    0.8148   0.8288   0.8273   +0.0125
v7nontear_span_s2026  0.8160   0.8282   0.8278   +0.0118
v5blend_span_s2026    0.8113   0.8258   0.8254   +0.0141
v6tears_span_s2026    0.8116   0.8235   0.8228   +0.0112
```

**The order does not change on either corrected ruler.** `band_span` — the shipping default, v2 — leads
on all three. The confound was real and it is now measured: correcting the ruler is worth +0.011 to +0.014
to every cluster-trained member against +0.001 to +0.003 for the v2-trained ones, exactly the direction
the confound predicts. It closes the gap from 0.0148 to 0.0034. It does not close it.

## Why this is the STRONG form, not a near miss

`v2delonly_span` is the right member to hang this on. Its arm pre-registered the outcome before the run
("On the v2 ruler: BELOW `band_span` ... On the corrected T1 ruler: at or above every other member in the
chain"), it corrects 2,080 cells against v6tears' 1,379 and v7nontear's 703, and it is the only table in
the family with a direction-stratified adjudication behind it. Half its prediction landed: it IS above
every other challenger on both corrected rulers, moving from fifth on T0 to second.

The other half did not, and T2 is why that matters. **T2 applies to the ruler exactly the operation
`v2delonly` applies to its training table** — the same selector, the same direction, the same 424 of those
cells that reach the 909. The ruler was therefore tilted toward this challenger by construction, and the
challenger still lost by 0.0034. A member cannot ask for a friendlier ruler than one built from its own
correction, so there is no further ruler to appeal to.

What T2 also retires is the suspicion that raised it. The broken addition half turns out to carry almost
nothing: 45 of the 469 reference cells, worth 0.0013 of `band_span`'s move and 0.0015 of `discord_mask`'s.
So T1 was half-broken and it did not matter — a null, and the cheapest kind, because it means every T1
number already on disk stands.

## Consequence

The labels axis is CLOSED at the shipping corner, on a ruler that is not v2, including one built from the
challenger's own correction. It cost no slot and no card. Re-opening it needs a label set that is not in
this list of eight AND a mechanism argued in advance — not another re-reading of these eight, which now
have three rulers and agree on all three.

Composition work should look elsewhere: `[[head_lr-at-the-shipping-corner]]` (knee-xuarh) is where that
search landed.
