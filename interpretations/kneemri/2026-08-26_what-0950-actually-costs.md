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
224 px** (forum 735304, recorded in `research/deep_dives/2026-08-25_competitor-single-model-scores.md`). Ours is
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

- ~~**RAD-DINO** (`knee-fh31`) — pre-registered directional band 0.856–0.865 = **+0.005 to +0.014
  offline**. Alone sufficient at its upper half.~~ **RAN AND REFUTED, 2026-08-26.** The cold re-run
  converged (12 of 12, peak 0.8141 at e7, then a clean four-epoch decline — not a truncation floor) and
  read **−0.0371 against `base_dense_s2026`'s 0.8512**, six times the paired floor, with the
  initialisation weights as the only variable. It was the biggest single entry in this stack and it is
  gone; nothing else here has a pre-registered band reaching +0.010 alone. **Do not re-queue it.**
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

## Update 2026-08-26 13:2xZ — the stack has no named member left, and that is the finding

Two hours after this page was written its own inventory has emptied from the top. RAD-DINO — the only
bullet whose pre-registered band reached the required **+0.010 offline on its own** — ran to convergence
and read **−0.0371**. `blendthree` tested the other route on the board and read **0.939 flat**. So both
of the two things this page named as capable of closing the distance have now been measured, and both
returned nothing.

**What survives is not a lever but a search.** `knee-s9cr` (P0) is the honest form of the remaining plan:
the two best self-reported singles on the board (0.942 and 0.938) are both at **224 px**, and the 0.942
was reached from 0.934 by unnamed *"tweaks"*. Our own `native224` prices 224 px at **−0.0023 against a
0.0062 floor** — flat — while costing **94 s/epoch instead of 290**. An unnamed tweak is found by trying
many, and 224 px is what buys many: **~19 min an arm instead of ~40**, roughly **2.3× the attempts per
free card hour**, at a resolution cost our own ledger measures as zero.

That reframes the arithmetic on this page one last time. The distance is still **+0.010 offline on one
member**, and it is now to be *searched for* rather than spent on a named bet — which means the scarce
resource stopped being submission slots (four spent, one held, none available until 00:00Z) and became
**card hours and attempt count**. Every exploratory arm from here runs at 224 px unless it is a
single-variable measurement against a 336 px control that already exists.

The four queued dense-corner arms stay at 336 as written: each carries a pre-registration against
`band_dense_hold`, and moving them would throw the control away to save twenty minutes.

## Update 2026-08-26 15:3xZ — the page's central arithmetic is wrong, and a seat measurement says so

Every number above rests on one equation, stated on line 34 and never tested:

```
ensemble  =  best single  +  bonus (+0.0125)
```

It says a member gain passes through to the ensemble **at 1.0**, so +0.010 offline on one member buys
+0.010 offline on the ensemble and, at `board_transfer` +1.72, first place. That is the assumption the
whole page is built on, and this project has now measured it.

`calib_dense_hold` is the best offline single member this repo has ever produced — 0.8634, **+0.0056**
over `band_dense`'s 0.8578 at the same corner, replicated in sign at `calib_span`. Priced into a seat by
`kneemri.kaggle.seat_value --against gradeswap_v1` (0.933 board, 7 members, every one of them scoreable —
the pinned champion cannot be priced at all because five of its six members are `_ff` full-fit):

```
roster of 7:                              0.8694
  + calib_dense_hold                      0.8704   (+0.0010)
  + band_dense_hold                       0.8695   (+0.0001)
best seat each could take:
  calib_dense_hold for grade_ladder:      0.8707   (+0.0013)
  band_dense_hold  for grade_ladder:      0.8696   (+0.0002)
0 of 2 clear +0.0015, under which this ruler has never been shown to pick a board tick
nearest neighbour: calib_dense_hold 0.9273 (mean 0.8925) · band_dense_hold 0.9119 (mean 0.8885)
seated spread: 0.8801 – 0.9248
```

**+0.0056 solo became +0.0013 in a seat. Pass-through is 23%, not 100%.**

And the mechanism is legible in the same block: `calib_dense_hold`'s nearest seated neighbour is **0.9273**,
higher than the entire seated spread's top (0.9248). Our best member is also our **most redundant**
candidate. That is not a coincidence — it is what "better model, same question, same supervision, same
pixels, same pooling" produces, which is
`2026-08-24_architecture_diversity_is_closed_and_the_empty_cell_is_strong_and_distant.md`'s mechanism
arriving from the other direction. Raising a member's quality moves `mean member error` down and takes
`ambiguity` down with it.

**What this reprices.**

| claim on this page | at 1.0 pass-through | at 0.23 |
|---|---|---|
| +0.010 offline single ⇒ first place | 0.950+ | ≈ +0.0023 ensemble ⇒ **0.943** |
| `stack3` (`knee-nu1m`), +0.0106 if every term is real and additive | 0.951 | ≈ +0.004 board ⇒ **0.943** |

So the ceiling of the entire queued member-quality lane is **0.943, not 0.950** — which is, exactly, the
number the owner named as enough for today, and is not first place. This does not kill the lane; it
prices it honestly for the first time.

**And it explains the flat band.** Ten board entries at 0.936–0.939 spanning count, member identity, fold
status, encoder family, vote rule, serving views, corpus and supervision are what a 23% pass-through looks
like from the outside. Nothing was wrong with those entries. The seats were being filled with better
answers to the same question.

**What survives, and it is one thing.** The board sequence 0.925 → 0.928 → 0.929 → 0.930 → 0.931 → 0.932 →
0.933 → **0.939** has exactly one jump larger than a tick, and it is `blendseat` at **+0.006**. Five of its
six members are `_ff` — full-fit, trained on fold 0 as well, +20% data. In the same window `blendthree`
swapped three seats for better-*supervised* twins and read **0.939 flat**. Same board, same six seats:
supervision bought zero, data bought +0.006. Data is also the only axis in this ledger that has ever
cleared the significance bar in either direction (`half_data` −0.0234, `quarter_data` −0.0671, a concave
curve whose next doubling extrapolates to ≈ +0.009).

**The three caveats, stated before the consequence.** (1) `n = 1` seat measurement of the pass-through, on
one pool, against one board entry. (2) `gradeswap_v1` is 0.933, not the 0.939 champion, so the pool priced
against is one board tick weaker and possibly less saturated, which if anything makes 23% an
**over**-estimate. (3) The `_ff` route reads on the board only — a full-fit member has seen the holdout, so
it cannot be scored offline at all, which is why the +0.006 was never explained until now.

**The consequence for the queue.** `stackfit` (`knee-orjq`) is still worth its card: it tests additivity,
which is a fact about the method and is what licenses or kills the whole stack idea. But `stack3` is no
longer a route to 0.950 and must not be sold as one — and it is the *wrong* lever for the ensemble even if
every term in it is real, because it is `calib_dense_hold` plus more flags on the same model, so its
nearest neighbour will be ≥ 0.927. More redundant, not less.

The route to 0.950 is not on this page any more. It is either data (`_ff` refits, the blocked external
corpora) or the empty cell — a member whose own loss is asked to produce ambiguity rather than hoped to.

---

## Correction 2026-08-26 16:0xZ — 23% is the MARGINAL pass-through. The uniform one is 92%, and it is a route

The section above is right about what it measured and wrong about what it concluded. It measured the
seat value of adding **one** better member to an already-saturated pool of seven, found +0.0056 solo
became +0.0013 in a seat, and wrote the ratio as if it priced member quality in general. It does not. It
prices exactly one case, and the decomposition the same section quotes says which:

    ensemble error  =  mean member error  −  ambiguity

Adding a member moves **both** terms. A better member lowers mean member error; a better member that is
also more redundant — and `calib_dense_hold`'s nearest neighbour is 0.9273, above the top of the entire
seated spread — lowers ambiguity by about as much. They cancel, and 23% is what survives.

Lifting **every** member with the same flag moves only the first term. The pool's mutual distances are
unchanged, so ambiguity is untouched and there is nothing to cancel against. That predicts a
pass-through near 1.0, and unlike the argument, this is now measured — three matched pairs, offline,
zero card:

| pool | members | holdout macro AUC |
|---|---|---|
| calibrated trio | `calib_dense_hold` + `calib_span_hold` + `calib_span` | **0.8685** |
| uncalibrated twins | `band_dense_hold` + `band_span_hold` + `band_span` | 0.8629 |
| | delta | **+0.0056** |
| | mean solo delta of the same flag at the same corners | +0.0061 |

**92%.** Both numbers are true and they answer different questions. 23% is what one more member is worth.
92% is what lifting the ones already seated is worth.

**What that repairs.** The page's closing line — "the route to 0.950 is not on this page any more" — was
written from the 23% number alone and is too strong. The member-*quality* lane is dead as a route only in
its **marginal** form: one more candidate, however good, into a saturated six. Its **uniform** form is
alive and is arithmetic rather than hope. Five of the champion's six seats are uncalibrated
(`calib_span_ff` is the sixth and already carries the flag):

    5 seats × 0.92 × +0.0061  =  +0.0047 offline
    × board_transfer +1.72     =  +0.0081 board
    blendseat 0.939            →  0.947

That is the `calibsix` arm (`knee-1c00`, `experiments/arms/calibsix.sh`), chained behind `margin_calib`:
seven hours of otherwise-idle overnight card, zero submission slots to produce, one to read.

**What it does not repair.** 0.947 is not 0.950, and no single queued arm closes the last three points —
`margin_calib`'s empty-cell seat (`knee-myl0`, +0.0018 offline ≈ +0.003 board if it lands) is the only
candidate, and its own pre-registration gives it 0.45. The three caveats above still stand unchanged, and
a fourth belongs with them: `blendthree` swapped three of six seats for better-**supervised** twins and
read 0.939 dead flat. That is the nearest precedent to `calibsix` and it returned nothing. The defence is
size and not kind — blend supervision measured +0.0034 at three seats, against +0.0061 at five here — and
if `calibsix` also reads flat, the uniform-lift route is refuted at the operating point and the ensemble
axis is closed for good rather than only for marginal members.

### Correction to the correction, 2026-08-26 16:0xZ — 92% was my own denominator error, and the truth is above 100%

The table above quotes "mean solo delta of the same flag at the same corners +0.0061". That is not a
mean. It is the **dense pair alone**, generalised to three pairs by nobody but me, and it is the
`CLAUDE.md` denominator failure in its purest form: I averaged over the corners that agreed with the
story. The ledger caught it — `experiments/README.md:318` records `calib_span` at **−0.0020 against
`band_span`**, a negative the "+0.0061" silently absorbed.

All six members re-scored one at a time over the same 909 shared studies:

| corner | calibrated | uncalibrated | Δ |
|---|---|---|---|
| dense, hold | 0.8634 | 0.8573 | +0.0061 |
| span, hold | 0.8565 | 0.8493 | +0.0072 |
| span, no hold | 0.8545 | 0.8565 | **−0.0020** |
| | | **mean** | **+0.0038** |

And the ensembles, on two nested pools:

| pool | calibrated | uncalibrated | Δ ensemble | Δ mean solo | pass-through |
|---|---|---|---|---|---|
| trio | 0.8685 | 0.8629 | +0.0056 | +0.0038 | **147%** |
| hold-only pair | 0.8679 | 0.8595 | **+0.0084** | +0.00665 | **126%** |

**The pass-through is above one, on both pools.** That changes the mechanism claim rather than just the
digit. A pure error-term lift predicts ~100% and I predicted 92%; measuring 126–147% means the
calibrated members also disagree with each other *more* than their uncalibrated twins do, so
`ambiguity` moved the helpful way instead of standing still. Calibration buys diversity as a side
effect. That is a better result than the one I claimed, arrived at by being wrong about the arithmetic.

**What it does not buy is confidence.** Every solo delta above is inside the 0.0062 three-seed floor,
the negative one included; only the +0.0084 pair clears it. And the confound is unflattering: both
positive pairs carry `--hold-annotations`, the negative one does not, and **`calibsix`'s five seats
carry none**. `calib_span` and `calib_span_hold` sit on the identical corner and differ in nothing but
that flag, yet their calibration deltas are −0.0020 and +0.0072. At n=1 per cell a 0.0092 spread
between two single-run differences is what scatter produces, and the ledger's own `calibrated_targets`
row is +0.0038 with no hold at the 0.2–0.8 corner — so no-hold is not uniformly negative. But it is
named as risk 4 in the arm, and if `calibsix` reads flat it is the first suspect, ahead of the
mechanism.

**Repriced, deliberately at the low end** (pass-through 1.0, not 1.3): 5 × 0.0038 = +0.0038 offline,
×1.72 = +0.0065 board, **0.939 → 0.946**, with 0.949 the optimistic end. The earlier +0.0047/0.947 was
right by coincidence — the mean solo fell and the pass-through rose by almost the same factor.
