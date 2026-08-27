# The foreign feed shortfall, and what is left after every lane closes

2026-08-27. Board: rank 35 of 2516, tied at 0.941, leader 0.952, frontier flat (top-5 +0.00000/day),
ours +0.00338/day. Four slots spent today (calibsix 0.941, calibmix 0.941, pixgroup 0.940,
honest12 0.937); one held.

## What was measured

`kneemri.evaluation.foreign_gate` reads a published Raptor checkpoint through one of our pixel caches
and scores it on the same 58 radiologist-annotated studies its own `gold_auc` was measured on. Their
corpus is 4,349 = 4,407 − 58: they withheld exactly the studies we hold out, so the two numbers meet
on one ruler. `coatnet_v4_full` publishes 0.9167.

| our cache key | macro | shortfall |
|---|---|---|
| `336px_20sl_0.1-0.9band` | **0.8819** | −0.0348 |
| `336px_20sl` | 0.8773 | −0.0395 |
| `336px_16sl` | 0.8727 | −0.0440 |
| `448px_12sl` | 0.8718 | −0.0450 |
| `336px_12sl` | 0.8700 | −0.0467 |
| `224px_12sl` | 0.8665 | −0.0503 |
| `224px_12sl_adjacent` | 0.8419 | −0.0749 |

Three things decompose out of it. **Resolution is not the cause**: at fixed 12-slice spacing the whole
224→448 span is +0.0053 and monotone, an eighth of the shortfall — 448 downsamples to their 384 and 224
upsamples to it, and neither direction costs much. **Depth sampling is the larger term**: 12→16→20
slices at fixed resolution buys +0.0027 then +0.0046, more than the entire resolution axis, and the
increments are not flattening. **Adjacent spacing is actively bad**: −0.0246 against spread spacing at
the same resolution, which agrees with Raptor's own `finespacing` archive (0.9031) losing to
`widedense` (0.9167). Band trimming to 0.10–0.90 is worth a further +0.0046.

The best key is `336px_20sl_0.1-0.9band` — which is our own dense-family training key. Our feed, at its
own best corner, serves their model 0.035 below what their feed serves it.

## What the shortfall is not

It is not a data lever for our own members. The ledger closed that axis from both ends before this was
measured: `knee-ezz` read 20 cached slices against 12 under max pooling at **−0.0001**, `extended_slots`
added nine slots and 83,297 unseen slices for **+0.0002**, `band140` closed the field-of-view axis, and
`attend_true` closed the pooling operator at **+0.0003**. The ledger's own words: the slice-aggregation
stage is saturated from both ends at once, and *the limit is upstream, in what one slice's feature
vector contains*.

So the curve prices one thing only: whether a foreign checkpoint is worth serving. It is a fact about
how much a model loses when it is fed pixels it was not fitted on, not a fact about our pixels.

## The member gap was a two-ruler artefact, and it is gone

`knee-3uje`: `calib_span_hold` reads **0.9152** on the gold 58 against `coatnet_v4_full`'s **0.9167**.
Our member is level with theirs. The 0.017 "member gap" of `knee-rh4t` and the 0.028 of `knee-xuj5`
were our holdout compared against their published number — two rulers, and the difference between the
rulers, not between the models. Macro standard error on 58 studies is about 0.012, so 0.0015 is nothing.

That removes the reason to want a foreign member at all. Served through their feed it is a peer;
served through ours it is a 0.88 member, which by our ledger is weak.

## And the seventh seat is inert anyway

`knee-97ib` measured it six times on the board:

| entry | board | seventh seat |
|---|---|---|
| `dark6_v1` | 0.937 | — |
| `dark7_v1` | 0.937 | `small_batch6_ff`, the best member we own (0.8598) |
| `lastseat_v1` | 0.937 | `small_batch6_s2026` |
| `farseat_v1` | 0.937 | `vjepa_warm_s2026`, the most distant member available |
| `augseat_v1` | 0.936 | `aug_long_s2026` |
| `allfit_v1` | 0.937 | `band_dense_long_s2026` |

Five different sevenths spanning best-quality to most-distant, one tick of spread. Adding a member to
a saturated roster does not move this board. `champion.json`'s pixel-distant 0.932 belongs to a
smaller, weaker roster where the marginal seat still had room; it does not license the move here.

The one thing that does survive from `knee-97ib` is its own prescription: **a new member is worth a
slot only as a swap into the six, or alongside a fusion change.** `votefit_v1` (0.938) is the only
entry that ever cleared the dark six, and it moved the fusion, not the roster.

## The arithmetic, stated before the measurement

Pass-through of a new member's solo quality into an ensemble seat measured **23%** (calibsix kernel,
2026-08-26). `board_transfer` is **+2.14** board points per holdout point. So a swap must show +0.0042
offline to be worth +0.009 board, and at 23% pass-through that needs a solo delta of +0.018 over the
member it replaces. Our member is level with theirs on the matched 58, and theirs reads 0.8819 through
our best feed. The arithmetic does not support the swap. It is measured anyway, because
`record_mix` is pure arithmetic over holdout arrays already on disk and costs no slot — but the slot
is not pre-committed to it.

## Label noise is half a ceiling, not a ceiling

`supervision_gap` on the gold 58: **labels 0.8658, model 0.8392**. The model is 0.0266 *below* its own
labels on aggregate, so label noise is not the binding constraint at the aggregate. The split is
six-and-six. The reader is the weak half on Effusion (+0.1304), Contusion (+0.0891), Synovitis
(+0.0508), Medial OA (+0.0419), Baker's (+0.0226) and Fracture (+0.0160); the labels beat the model on
Lateral Meniscus (−0.1925), ACL (−0.1225), MCL (−0.0964), Medial Meniscus (−0.0938), PF OA (−0.0856)
and Lateral OA (−0.0793). Denominator: 58 studies, 696 cells — per-finding AUCs at that n are not
quotable, and only the direction of the split is.

`blendlabels` already priced the lane at +0.0034 on the mean and +0.0070 on exactly those six
label-bottlenecked findings, and the ledger's verdict stands: live, small, an order of magnitude short
of the bar, not the path to 0.950.

The Raptor download carries a fourth reader (`labels/labels_llm_soft.csv`, 4,349 studies, unmentioned
level 0.05 against our 0.28). It cannot be scored on the gold 58 at all — those 58 rows are exactly the
ones absent from their corpus, so every gold cell reindexes to NaN and the tool reports a flat 0.5000
with 0% at every cut. That is a broken denominator, not a null. And `table_separation` has already
bracketed four blends spanning 0.8894–0.8931 with every pairwise interval covering zero, so a fourth
reader is unresolvable before it is built.

## The honest state

Every lane the ledger has measured is closed or small: decode geometry (~+0.005 in total), pooling
operator (+0.0003), slice count and slots and field of view (all ≤ +0.0002), vote rule (0.0005 spread
across three rules), the seventh seat (inert six times), full-fit refit at the marginal seat (0.000),
corpus merge (does not carry to the dense corner), label blending (+0.0034), the calibration uniform
lift (delivered +0.002 board against a +0.0065 prediction, landing in its own arm's *"one tick, inside
board scatter"* band).

0.941 → 0.950 is +0.009 board. No measured lever on this list produces it, and the frontier above us
has not moved in days, which says the same thing from the other side. The rational use of a last slot
is the highest-expected-value single bet with the honest expectation attached — +0.000 to +0.003 — not
a claim that the gap closes today.
