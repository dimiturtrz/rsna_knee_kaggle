# The holdout-to-board transfer is ~1:1 at our level, and it compresses as we climb

**Date:** 2026-09-05 · **Card spent:** none · **Slots:** none · **Board read:** 2026-09-05T11:08:57Z

Rank 40 of 3105, ours 0.944, leader 0.954, gap **0.010**. `board_read --live` also prices the trend:
ours +0.00036/day, the top five +0.00025/day, closing **+0.00011/day** — at which rate the gap shuts in
**70 days**. Whatever the deadline is, that is not a plan. Only a step change reaches 0.953.

So the question is what a step change has to be worth in the currency we can actually measure, and that
is the transfer constant. Read from `board_transfer` rather than quoted from prose, because it has now
been quoted wrong from prose twice:

| scope | resolved pairs | median board points per holdout point |
|---|---:|---|
| the whole record | 395 | **+1.13** — the module's own text calls this "the wrong number to price an arm with" |
| entries ≥ 0.8670 (where we live) | 17 | **+0.98** [+0.94, +1.17] |
| entries ≥ 0.8695 | 2 | **+0.80** [+0.71, +0.89] — DIRECTION ONLY |

**The transfer is about 1:1 at our operating point, not the +1.55 that `kneemri/kaggle/board_check.py`'s
docstring carries** (it records "+1.82 against a complete set's +1.55" from the 2026-08-25 repair, when
`comp/board.jsonl` was found holding 34 of 58 submissions). That constant is a whole-record fit, and the
whole record is exactly the scope this module refuses to price an arm with; at our operating point it
overstates what an arm buys by roughly half. The top band is two pairs and cannot price anything on its
own, but it points the same way as the whole-record-to-operating-point step does: **the ruler compresses
as the score rises.** That is the ordinary shape — the board's own headroom shrinks, and the pairs that
survive at the top are the ones where the holdout gap is largest and least representative.

## What it costs to close 0.010

At +0.98 the gap needs **+0.0102 of holdout ensemble AUC**. If the +0.80 band is real at the top, it
needs **+0.0125**. Against a 0.0062 holdout floor and a ~0.015 result bar, that is **1.6 to 2 results**,
not one. Nothing currently on the card or in the pool is priced that large:

- the per-label axis, re-priced today, is **+0.0159** of board on MCL + Lateral OA — and every mechanism
  under it was closed card-free this morning, so the number is a statement about what those columns are
  worth, not about any operation that reaches them
- decode and architecture axes are closed at about **+0.005 total** (`README:4627`)
- the member/seating axis is closed on seven board flats
- the crop route, priced today, is **+0.0007**
- **the EPOCH BUDGET is not binding, closed card-free at 12:54Z.** "Twelve epochs is capping us" is the
  cheapest ≥0.010 story available — if runs were peaking on their last epoch, more schedule would be a
  mechanism rather than a knob. `convergence_check` over the whole corpus says no: **212 runs, 165
  converged, 14 truncated, 3 stopped, 8 undertrained.** Seven of the eight undertrained asked for only
  four epochs and are short probes (`adapt_null`, `mae_matched`, `raptor_feat_pre`, `small_feat_pre`,
  `swin_distill_pre`, `lora_r8_small_336`, `wholestack_fold2`). Exactly ONE twelve-epoch run peaked on
  its last epoch — `encoder_adapt_lower_s2026`, e11 of 12. One run in 165 is not an axis. Extending the
  schedule buys nothing, and the reason to write that down is that it costs card to re-ask.
- **the corpus SITE-COUNT corner is negative, priced today off an arm nothing had read.**
  `merged3_corpus_s2026` ran on 2026-09-01, converged (e9 0.8511, e10 0.8510, e11 0.8510), and sits on the
  roster at **0.8521** with not one row in `experiments/README.md`. Its control is `merged_corpus_s2026`
  **0.8545**, one variable away — the same two-site recipe with MRNet added as a third site under the
  union table `merged3_labels.csv`. A third site is **−0.0024**. That is the measurement that prices the
  remaining corpora: fastMRI as a fourth site is the same operation one step further out, and the nearest
  reading of that operation is negative. The corpus axis `README:4627` calls "still moving" is the
  LABEL corner, not the site corner — `merged_blend_cf` 0.8597 and `merged_span_cf` 0.8618 both hold the
  two-site cache fixed and change what supervises it. The per-label read makes the mechanism plain and
  is worse for the site corner than the mean is: MRNet was argued onto the schedule **on KIND**, because
  it labels meniscus and Rijeka does not, and Lateral Meniscus came back **−0.0217, last of twelve** —
  while Synovitis, the one starred cost Rijeka's arrival left unexplained, came back **+0.0234**. Two
  foreign sites TRADE the same columns rather than stacking. A corpus mean can sit inside the floor while
  every column under it rotates, and adding site four has no reason to do anything else.

## The consequence for how card gets spent

An arm worth +0.005 of holdout is worth about **+0.005 of board** here, and about +0.004 if we get to
0.8695. Two of them do not sum to the gap, and the ledger's history says they do not sum at all —
substitution rather than addition is what this project keeps measuring (span vs depth regions, span vs
crop, corpus at the blend corner vs the dense corner).

So the only card worth spending is on a mechanism whose plausible size is **≥ 0.010 of holdout on its
own**. That is one class of thing at present: a trunk that is genuinely outside the DINOv2 family AND
reaches the seat band on its own. `fault_ruler` is why the second half of that sentence is not optional
— across a quality gap, member distance correlates **−0.5313** with the gain it is supposed to predict,
and **+0.7086** only with quality held fixed. RadImageNet is the worked counterexample: the most distant
candidate the pool has ever priced (nearest 0.4394 against a 0.8956 roster boundary) and worth −0.0004,
because its solo is 0.6716.

`mricore_corpus` is the live test of exactly that gate and is queued. Q-SOLO, with Kimi, asks whether any
other published trunk has a knee number in the seat band at all. If the answer is no, the trunk lane
closes on the gate and the step change has to come from the corpus or the target — which is what Q-TRUTH
is for.

**What must NOT be concluded:** this does not say the gap is unreachable. It says the exchange rate is
1:1 and falling, so a plan built out of +0.003 increments does not arrive, and pricing an arm at +1.55
would make three such increments look like the whole gap when they are worth two thirds of one.
