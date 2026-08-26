# What 0.950 actually costs, priced off the board rather than off ambition

2026-08-26 · board read 2026-08-26 06:0xZ (rank 47 / 2418, leader 0.952)

The goal has been stated as a score for two weeks and never once as a *quantity*. This prices it, so
that the next arm is chosen against a number instead of against momentum.

## The record it is computed from, and its denominator

`comp/board.jsonl` holds **65** rows. **47** carry `scope: ensemble` with a board score; **10** are
`scope: excluded` (priced out, with the reason); 4 are `member_mean` and 4 unscoped. Of the 47, exactly
**three** are single-member entries (`members: 1`), and only **two** of those have an ensemble
submitted within a day of them — so every bonus figure below rests on **n = 2 pairings**, and that is
the honest sample, not a rhetorical hedge.

## The ensemble bonus, recomputed

| date | best single | best ensemble that day | bonus |
|---|---|---|---|
| 2026-08-17 | `solo_v4` 0.914 | `solotop_v1` 0.925 | **+0.011** |
| 2026-08-26 | `soloseat_v1` 0.925 | `blendseat_v1` 0.939 | **+0.014** |

(`solo_v1` 0.868, offline 0.8518, has no contemporaneous ensemble and is excluded from the pairing.)

`knee-urvf` states the bonus as spanning **+0.005 to +0.026**. On the two clean best-vs-best,
same-window pairings it is far tighter: **+0.011 to +0.014, mean +0.0125.** The wider span is a
looser pairing. **Take the tight one for planning and the wide one for risk** — the difference between
them is the entire question of whether the ensemble axis still has room, and n = 2 cannot settle it.

## The number

Ensembling nine entries has now produced 0.936–0.939 — **flat**. Under the tight bonus:

```
target ensemble        0.950
  − mean bonus         0.0125
  = required single    0.9375   board
  − our best single    0.925    (soloseat_v1)
  = the gap            +0.0125  board
  ÷ board_transfer     1.72
  = required OFFLINE   +0.0073
```

And against the frontier rather than against ourselves: a rival reports a **0.942 single model at
224 px** (forum 735304, recorded in `research/2026-08-25_competitor-single-model-scores.md`). Ours is
0.925. That gap is **+0.017 board = +0.010 offline**, and closing *it alone* — changing nothing else,
keeping the same six seats and the same vote — lands the ensemble at **≈0.954**, above the current
leader.

**So the whole remaining distance to first place is single-member quality, and it is +0.010 offline.**

**Why the offline figures go through `board_transfer` and not through a holdout.** `soloseat_v1`'s
member is `final_ws_e12`, which is **full-fit** — trained on fold 0 as well — so its 0.9541 fold-0
reading is contaminated and is NOT this project's best offline single (`experiments/README.md:3861`
already excludes it from `seat_value` for that reason). There is exactly one row on the board carrying
both numbers: `solo_v1`, offline 0.8518 → board 0.868. Every offline figure on this page is therefore a
CONVERSION at +1.72 board points per offline point, not a measurement, and inherits that constant's
error. Quote the board numbers; treat the offline ones as targets.

## Why that reframes what counts as worth doing

+0.010 offline is **0.67× the 1.5% significance bar** and **1.6× the 0.0062 paired noise floor**. Two
consequences, and they pull against each other:

1. **First place does not require a finding.** It requires a *stack* of directional gains. `CLAUDE.md`
   already licenses exactly this — *"increments below the bar still compose, and a chain of them that
   sums past it IS a result"* — and this is the first time the arithmetic has said the chain is the
   ONLY route rather than a consolation for failing to find a lever.
2. **No single arm in that stack will be individually believable.** A +0.006 member gain sits at the
   noise floor; one seed cannot distinguish it from scatter. The discipline that follows is to keep
   directional gains **when free and mechanism-sound**, and never to spend a confirmation seed on one —
   which is the standing rule, now with a reason to obey it that is about winning rather than about
   honesty.

## What is in the stack, and what it is blocked on

- **RAD-DINO** (`knee-fh31`) — pre-registered directional band 0.856–0.865 = **+0.005 to +0.014
  offline**. Alone sufficient at its upper half. The 2026-08-23 reading is void twice over (mis-fed
  normalization + killed at epoch 7 of 12, `knee-6477`), so this is a cold re-run, ~3.3 h, reads on a
  fold-0 holdout, **costs no slot**. Card-blocked: celltrack holds it 5–7 h from ~09:20Z.
- **blendsix** — five members refitted on harmonised labels and geometry. This is the *other* route:
  it raises the FLOOR of every seat, which is `knee-urvf`'s own prescription for moving an ensemble.
  All five are `--fold -1`, so it reads **only on the board**, i.e. not before the 00:00Z reset.
- **Untried recipe axes** — mixup (`knee-x580`) was the one never attempted. It has since been run and
  **closed in its own band 2**: peak 0.8505 vs the matched control's 0.8490 (**+0.0015, a quarter of the
  noise floor**), while the peak→finish slide halved, 0.0088 against 0.0165. The regulariser fires — final
  train loss 0.2 higher, peak three epochs later — and the ceiling does not move. It contributes **nothing**
  to the stack; it is a free rider for long-budget arms. Its real yield is a separation: the slide is
  memorisation and is curable, and curing it returns +0.0015, so **the slide is not the bound**.
- **Owner-blocked** — MRNet (`knee-ombx`), fastMRI+ (`knee-3oy8`). Both attack the representation gap
  (`knee-jk3`, 0.044 at matched train loss) that four closed lanes — resolution, crop geometry, the
  label axis, laterality — have now left as the only live suspect.

## The one thing that would falsify this page

The bonus is fitted on two points. If it is really +0.026 at the top of its span, then a 0.924 single
inside a better-diversified ensemble reaches 0.950 with **no member improvement at all**, and blendsix
is the right bet rather than a floor-raiser. If it is +0.011, the ensemble axis is spent and only the
member matters. **blendsix's board reading is the experiment that separates those two worlds**, and it
is worth a slot for that reason and not for its score.

## Update 2026-08-26 06:24Z — the floor-raise was tested at half strength and bought nothing

`blendthree_v1` (ref 55787345) is that experiment at **dose 3 of 6**: the champion's exact six seats,
vote rule, count, kit and serving config, with three seats swapped for their own blend-supervised twins
on harmonised labels and 12/0.2-0.8 geometry. It read **0.939 — the champion's own number, to the tick.**

That is the first DIRECT board test of `knee-urvf`'s floor-raise prescription, and raising the floor of
**three of six** members moved the ensemble by zero. It does not refute dose 6 — a linear dose effect
reads flat at the midpoint, which the pre-registration said before the number — but it lowers its prior,
and it is the tenth entry in the 0.936–0.939 band, now spanning count, member identity, fold status,
regularisation, encoder family, architecture family, vote rule, serving views, corpus and supervision.

**The page's own falsifier now points one way.** Ten flat readings are not what a +0.026 bonus looks
like; they are what a spent axis looks like. Planning weight moves further onto the **tight** bonus
(+0.0125) and therefore onto the member: the remaining distance to first place stays **+0.010 offline
on one member**, and the ensemble is not going to be talked into supplying it.

The dose-6 endpoint is still the clean separator and is still worth a slot — but slot 5 is held by owner
directive (`knee-pubw`), which beats the bead's own firing rule. It waits for 2026-08-27, and it is asked
for before it is spent.
