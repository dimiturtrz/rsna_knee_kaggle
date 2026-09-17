# In-plane token scale lifts fluid and soft tissue, and is FLAT-to-negative on the thin bright lines it was proposed for

2026-09-17. Bead knee-e9f83. No card spent: a read of one pair already on disk, taken before a card was
committed to the axis.

## The lever as it was ranked

The corner's residual label gaps — MCL, Lateral OA, Fracture, each ≥0.11 below their perfect-truth
ceilings — are all **thin, bright, high-frequency structures**: a ligament band, a joint-space margin, a
cortical break. DINOv2 at 336px over a 14px patch grid sees each of them across very few tokens. The
mechanism was that raising in-plane token density is the one axis that could put more tokens on exactly
those three, and it was ranked first among open levers at roughly 1.5%-class.

It costs a cache rebuild before it costs a card: the largest cache on disk is 336px, so there is no
448px array to train against.

## The cheap gate, run first

`recipe_pairs --key image_size` over the resolved recipes finds exactly **one** isolating `image_size`
pair on the main population:

    224  0.8718  ->  336  0.8781   +0.0063 CLEARS   band_deep224_s2026 / band_deep_matched_s2026

A 1.5× linear step that barely clears the 0.0062 floor. Extrapolating linearly in token count, 336→448
is a 1.33× step and prices at roughly **+0.004** — under the floor, for 3–5 h of card plus the rebuild.
That alone would have made it a poor ratio. The per-label read is what actually kills it.

## The per-label read, which is the decisive part

Same pair, same two runs, broken out by label:

| delta | label | 224 | 336 |
|---|---|---|---|
| **+0.0314** | Synovitis | 0.7841 | 0.8155 |
| **+0.0280** | Baker's | 0.8709 | 0.8988 |
| +0.0127 | **MCL** | 0.9405 | 0.9531 |
| +0.0086 | Lateral Meniscus | 0.6473 | 0.6560 |
| +0.0063 | PF OA | 0.7095 | 0.7158 |
| +0.0060 | Effusion | 0.8172 | 0.8232 |
| +0.0042 | ACL | 0.9644 | 0.9686 |
| +0.0000 | Contusion | 0.9680 | 0.9680 |
| −0.0027 | Medial Meniscus | 0.7914 | 0.7887 |
| −0.0030 | Medial OA | 0.8494 | 0.8464 |
| −0.0103 | **Fracture** | 0.9452 | 0.9349 |
| −0.0892 | **Lateral OA** | 0.7830 | 0.6938 |

The macro +0.0063 is a **sum of opposing moves**, not a lift. Of the three labels the mechanism was
built on, only MCL behaves as predicted; **Fracture and Lateral OA both go the wrong way**. What finer
tokens actually buy is the fluid and soft-tissue pair — Synovitis +0.0314 and Baker's +0.0280, five
times MCL's move — which are diffuse, bright, region-scale findings rather than thin lines.

So the mechanism is wrong in its own terms. More tokens is not "more resolution on hairlines"; on this
one reading it behaves like more capacity for *area* evidence, and the hairline labels are indifferent
or worse.

## What this is NOT

Three caveats, and they are load-bearing rather than polite:

- **n = 1 pair.** One comparison, not a curve. The 448 step is being priced off a single 224→336 read.
- **Non-corner recipe.** Both runs sit in the ~0.87 band, well below the 0.9053 corner, and this repo
  has been wrong before about a delta read on one recipe family transferring to another — that is the
  whole content of the `labels` family confound written up the same day.
- **Rare-label AUC is noisy.** Lateral OA's −0.0892 is far outside anything a resolution step should
  plausibly do and is likely mostly instability on a thin positive count. It should not be quoted as a
  measured harm; it should be read as "does not support the mechanism", which is all this file claims.

What survives all three is the shape, not the sizes: the labels that move are not the labels the lever
was proposed for, and the labels it was proposed for do not move up.

## Verdict

**Not armed.** ~+0.004 extrapolated at 3–5 h plus a cache rebuild is a bad ratio on its own, and the
mechanism that made it a 1.5%-class candidate does not survive its own cheapest test.

The axis is not closed — it is REDIRECTED. If in-plane token scale is ever run, the pre-registration it
should carry is **Synovitis and Baker's**, not MCL/Fracture/Lateral OA, and the honest band is a few
thousandths of macro. And the finding that the residual gaps point somewhere OTHER than in-plane
resolution stands as the useful half: see
`2026-09-17_labels_is_the_most_priced_leaf_and_the_corner_has_tried_three_schemes.md`, whose closing
paragraph pointed at this lever and is corrected by this file.
