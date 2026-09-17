# `labels` is the corpus's most-priced leaf, and the shipping corner has tried three schemes

2026-09-17. Bead knee-e9f83. No card spent: a read of `kneemri.experiments.recipe_pairs` over the run
records already on disk.

The owner's standing question, asked before this was looked at: *"you read past attempts in
interpretations right? because you have often said the labels are the lever and always stopped these
works in the end. check that you are not going to waste our time."* This is that check, and it was
answered before any card was priced.

## What the instrument says, unprompted

`recipe_pairs` prices every leaf of every resolved recipe against every other, on one holdout
population, guarded on population, truncation, schema drift and in-sample void. Over 283 quotable runs
it finds 103 single-key isolating pairs across 31 leaves. **`labels` leads by a wide margin:**

| key | pairs | clearing 0.0062 |
|---|---|---|
| **`labels`** | **30** | **17** |
| `variant` | 8 | 6 |
| `backbone_lr` | 7 | 4 |
| `calibrated_targets` | 6 | 4 |
| `epochs` | 6 | 3 |
| `encoder_weights` | 6 | 0 |
| `slice_pooling` | 6 | 1 |
| everything else | 1–3 | 0–2 |

Nothing else in the corpus has half as many resolved pairs. Read alone, that table says the label
scheme is the single most consequential choice this project makes — which is exactly the reading that
has been reached, and abandoned, before.

## Why that reading is a family confound

Every one of the seventeen clearing `labels` pairs is drawn between two `*_span_s2026` runs — the
**older recipe family**: `slice_pooling=max`, `slot_dropout=0.0`, `epochs=12`. That family's best run
anywhere is `confident_span_s2026` at **0.8785**. The shipping corner is **0.9053**.

So the label axis was swept thoroughly, on a recipe **0.027 below the one we ship**, and never swept
where we ship. The deltas are real — `confident.csv` really does beat `report_labels_v6tears.csv` by
0.0205 at 12 epochs with max pooling — but they are readings about a corner nobody is standing on.

## What the corner itself has actually tried

Ten quotable runs sit at `labelwise / 0.40 / 24`. Their label files:

| ledger75 | labels | scheme | cal | run |
|---|---|---|---|---|
| **0.9053** | `report_labels_v2.csv` | canonical | True | `labelwise_depth24_drop40_long_s2026` |
| 0.9051 | `report_labels_v2.csv` | canonical | True | `labelwise_drop_drop40_long_s2026` |
| 0.9048 | `report_labels_v2.csv` | canonical | True | `labelwise_dense_drop40_long_s2026` |
| 0.9045 | `report_labels_v2.csv` | extended | True | `labelwise_ext_drop40_long_s2026` |
| **0.9044** | **`report_labels_negpull.csv`** | canonical | True | `labelwise_depth24_negpull_s2026` |
| 0.9018 | `report_labels_v2.csv` | roinative | True | `roinative_labelwise_drop40_long_s2026` |
| 0.9007 | `report_labels_v2.csv` | canonical | True | `labelwise_dense_stem_llrd_s2026` |
| 0.8984 | `report_labels_v2.csv` | roi | True | `roi_labelwise_drop40_long_s2026` |
| 0.8982 | **`blend_three_v1.csv`** | canonical | True | `chainfour_drop40_long_s2026` |
| 0.8901 | **`mergedk_labels.csv`** | mergedk | False | `mergedk_labelwise_drop40_long_336_s2026` |

**Three alternative schemes, out of twenty-two the corpus owns.** The best of them, `negpull`, is the
one isolating pair here — `report_labels_v2.csv` → `report_labels_negpull.csv`, **−0.0009**, seven
times inside the paired floor. `blend_three` is −0.0071 and `mergedk` is −0.0152, and neither is
isolating (mergedk also moves `calibrated_targets` and `scheme`).

The size that matters is in the left column rather than in any of those deltas. **Seven runs at the
corner all read `report_labels_v2.csv` and span 0.9053 to 0.8984 — a 0.0069 spread produced by other
knobs entirely.** The best alternative label scheme moves the number by a seventh of the scatter the
same label file already shows across its own siblings. At this corner the label axis is not merely
flat; it is smaller than the noise the corner makes against itself.

## So: is it a lever?

Not as the pair table presents it, and that is the answer to the owner's question — **the corpus's
biggest-looking lever is a gradient across a recipe family we left behind.** Anyone reading the 17
clearing pairs as an invitation to sweep labels at the corner would be re-running, at 1.6 h a throw, a
comparison whose one honest instance already came back at −0.0009.

What is NOT settled, and is the whole of what remains:

- **Nineteen of twenty-two schemes have never run at the corner.** Three tried, all flat or worse, is
  weak evidence that the nineteenth is too — but it is evidence about a sample of three.
- **`confident.csv` has never been compared to `report_labels_v2.csv` at any recipe.** It is the span
  family's best at 0.8785 and it beat four sibling schemes there by 0.015–0.021, but the span family
  holds no plain-v2 run to pair it against, so the two have never met. That comparison does not exist
  anywhere in the corpus, at any recipe, which is different from having been made and come back flat.

Neither of those earns a card at 1.6 h against a 0.9053 control while the corner's own knob axis is
closed and the residual label gaps (MCL, Lateral OA, Fracture, each ≥0.11 below their perfect-truth
ceilings) look structural rather than textual. **This sentence originally named in-plane resolution as
where they point, and that was refuted the same day** — the one `image_size` pair on disk lifts
Synovitis and Baker's while Fracture and Lateral OA move the wrong way. See
`2026-09-17_in_plane_resolution_lifts_fluid_not_thin_bright_lines.md`. They are recorded here so that the
next reader who arrives at the 30-pair row arrives at this paragraph with it, instead of at the
sweep.

## The general shape

Third instance in two days of one defect, and the cheapest yet to spot because the shape now has a
name. The ledger compared numbers read on different study sets. The corner carried a rate fitted under
a different schedule. Here a leaf was swept exhaustively under a recipe family that has since been
replaced, and its deltas were inherited by a corner that never took them. In all three, nothing was
broken and every check was green; what was missing was a record of what the number is a statement
ABOUT. `recipe_pairs` now prints the two runs behind every delta on the same line, which is the
cheapest form that record can take.
