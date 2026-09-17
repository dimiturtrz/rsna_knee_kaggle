# One member equals eight, and the member's own family has saturated

Read 2026-09-17 off `comp/board.jsonl` and a full `ledger75` sweep of 442 members. Written because
every number below had been re-derived at least twice from scratch in the preceding two days.

## The board fact

```
eightseat_v1 | ref 56288513 | board 0.944 | members 8
swapseat_v1  | ref 56288717 | board 0.944 | members 7
ffsolo_v1    | ref 56296698 | board 0.944 | members 1  ['labelwise_depth24_drop40_long_ff']
```

Champion is 0.946. One member ties our seven- and eight-seat entries. The board reports at 0.001, so
the entire ensemble premium over a single good member is **<= 0.002, and against these two entries it
is exactly zero**.

**Consequence, and it is the load-bearing one: every redistribution lever is capped by that number.**
Seating, reseating, vote-rule changes and distillation-from-our-own-seats all move probability mass
between members we already have. None of them can pay more than the premium, and the premium is
inside the board's own resolution. The remaining gain is in the MEMBER, not in the committee.

## The member, and the plateau around it

Honest held-out members only (`raptor_*` and `foreign_oof_corpus` are excluded: `foreign_overlap`
measures them 98.6% in-sample, which is what voided `raptor_solo`'s 0.8938; `dinov3f0_solo` is DINOv3
and we ship DINOv2 only):

```
labelwise_depth24_drop40_long_s2026     0.9053   best epoch 18 of 24
labelwise_drop_drop40_long_s2026        0.9051
labelwise_dense_drop40_long_s2026       0.9048
labelwise_ext_drop40_long_s2026         0.9045
labelwise_depth24_negpull_s2026         0.9044
stackfit_s2026                          0.8962   best non-family solo
band_dense_s2026                        0.8877   the family's control
```

`ffsolo_v1` ships the full-corpus (`_ff`) twin of the top row. A `_ff` member holds nothing out, so no
ledger row can price it; 0.9053 is its fold-0 sibling.

Five variants of one recipe within **0.0009**. The family is at its ceiling. Nothing among 442 members
beats it.

## Both operators are already at their optimum

The two knobs that produced every winner we own are swept, and neither has headroom left:

```
slot-dropout 0.25   0.8873
slot-dropout 0.40   0.8877   <- shipped operating point, the peak
slot-dropout 0.55   0.8844
```
Span 0.0033, inside the 0.0062 paired floor. And `--epochs 36` peaks at **epoch 17 of 36** (0.9030),
so the 24-epoch budget is not binding either. Neither knob is an open lever.

## In-domain SSL is negative on all three objectives

Against the matched control `band_dense` 0.8877:

```
latent MIM       latent_register_dense_s2026   0.8619   -0.0258
latent MIM       latent_register_deep_s2026    0.8609   -0.0268
volume-native    vjepa_warm_s2026              0.8673   -0.0204
cross-corpus     mrnet_warm_s2026              0.8773   -0.0104
```

The fourth objective, spectral MAE with the unit fix (`mae_unitfix`, bead knee-zxm9z), has never been
fit to convergence. Its upstream mechanism gate PASSED — the unit fix moved pixel loss 0.9682 -> 0.5570,
3.2% -> 44% of hidden-patch variance explained — but **its three surviving fine-tune epochs read
0.6637 -> 0.6898 -> 0.6957 at e3 against its own control's 0.8232 at e3**. It is 0.128 behind at the
matched epoch and would have to close that AND add 0.015 in nine more. The bead's `P(clears 1.5%) ~ 0.25`
was written 2026-09-10, before the three readings above existed to bound it. **Re-price before arming;
do not spend 5.7 h on the 2026-09-10 number.**

## What this leaves

No open lever is priced above a knob. The honest statement is that the member recipe, the two operators
that tune it, the committee that votes it, and three of four SSL initialisations are all measured and
closed. Anything that pays 1.5% from here has to change something none of those touch.
