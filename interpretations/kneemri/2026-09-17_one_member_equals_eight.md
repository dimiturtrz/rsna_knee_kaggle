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

## Why one member ties eight: the members fail the SAME studies

`study_attribution --run labelwise_depth24_drop40_long_s2026`, 909 held-out studies, tail = worst 90:

```
tail overlap with 12 other members: 0.631 against a chance 0.100
studies in this tail and no other member's: 8
SHARED
```

Six times chance. Eighty-two of our ninety worst studies are also in some other member's worst
ninety. **A committee averages over disagreement, and on the studies that decide the score the
members do not disagree.** That is the mechanism behind the <= 0.002 premium above, and it is
why no reseating, vote rule or distillation from this pool can pay: they all redistribute a
disagreement that is not there.

## The residual is not a recognisable population, and it is not the labels

Same run. `study_attribution` classifies every tail and body cell by the report-reader failure
that could explain it:

```
missed_negation  tail 0.178  body 0.225   -0.047 +- 0.043
genuine          tail 0.100  body 0.076   +0.024 +- 0.033
unlocated        tail 0.700  body 0.646   +0.054 +- 0.051
categories separating tail from body at 2 sigma: NONE
```

And `population_split --records labelwise_depth24_drop40_long_s2026.holdout.npz` cuts the errors
by the only acquisition axes the competition ships (series count; fluid-sensitive count — plane
has one stratum and `Fat_Suppression` duplicates `Fluid_Sensitive`):

```
by series count       errors/positive   <=4 0.487   5 0.569   6 0.563   7+ 0.528
by fluid-sensitive    errors/positive    2  0.534   3 0.548   4+ 0.566
tail vs body          series 5.467/5.457   planes 3.000/3.000   fluid 0.600/0.594
```

Flat. There is no thin-acquisition lane and no label-noise lane, so neither the DATA lever
(sample more of a kind, weight it, render it differently) nor the LABEL lever (a better report
reader) has a target to aim at on this member.

## Three interrogations, three nulls, one shape

| axis | instrument | verdict |
| --- | --- | --- |
| by finding | `colhalf_v1` ref 56037952 (board) | weak six 0.932 vs strong six 0.956 against a 0.110 offline spread — no weak lane |
| by study population | `population_split` | flat across every shipped acquisition axis |
| by error cause | `study_attribution` | no category separates the tail at 2 sigma |

The headroom is DISTRIBUTED on every axis we can cut. Every one of the twelve columns has to
move about 0.010 of AUC, and the studies that move it are the ones every member already fails
together. That rules out targeted work as a class, which is a strong constraint and the most
useful thing on this page.

## MCL: the gold-set verdict rests on nine positives

Three numbers for one column, and they do not agree:

```
ledger, LLM report labels, 909 holdout studies   0.8104   (our WORST column)
supervision_gap, radiologist, 58 gold studies    0.8798   (9 positives)
colmcl_v1 ref 56020054, board ablation           0.980    interval 0.968-0.992
```

The board ablation is the only one measured against the competition's own truth at the
competition's own scale: null one column, watch the mean of twelve fall, multiply by twelve.
`colnull_v1` (all 0.5, board exactly 0.500) is what licenses that arithmetic. The
"MCL is MODEL-bound at -0.0964" reading is from the 58-study gold set with nine positives and is
underpowered against it. **Do not aim work at MCL on the offline number** — it is the single
widest offline-to-board disagreement the project has measured, 0.17.
