# Seat value tracks representation, and the board gap lives in eight columns

2026-09-11, late. Two findings measured by the researcher (D381, D382) and read here. The first closes
the lever this project has been spending nearly all its card time on. The second is the first lever
anyone has found that is the right SIZE for the gap, and it is the reason the first one closing is not
a disaster.

All numbers are macro AUC at `holdout909`, midpoint-binarised v2 truth, unless a line says otherwise. A
number in this repo is midpoint unless it says otherwise.

---

## 1. Solo value and seat value are different quantities, and seat value tracks representation

Every attribute marginal this project has ever fitted is a SOLO marginal: median-with minus
median-without, over members scored alone. The quantity that actually decides whether an arm is worth
running is its value as an added SEAT to the ensemble we ship, and the two had never been separated.

D381 separated them. Protocol: half-split, 40 readings, LogitAverage, macro AUC midpoint @holdout909,
population rebuilt across both roster shelves to 134 packaged-honest members, candidates drawn from the
0%-exposure sub-pool (n=125). For each attribute, the delta from adding the best member CARRYING it as
a seventh seat to the fixed honest six, against the best member NOT carrying it.

| attribute | solo marginal (0% pool) | SEAT delta | win/40 | times seated in the six |
|---|---|---|---|---|
| merged-rijeka | +0.0117 | +0.0005 +/- 0.0005 | 36/4 | 0 (licence-blocked) |
| labelwise | +0.0138 | +0.0005 +/- 0.0005 | 31/9 | 1 |
| calib-tgt | +0.0109 | +0.0002 +/- 0.0006 | 29/11 | ~4 |
| blend-three | +0.0109 | +0.0001 +/- 0.0005 | 18/22 | several |
| hold-annot | +0.0112 | -0.0000 +/- 0.0007 | 17/23 | ~4 |
| ema | +0.0105 | -0.0003 +/- 0.0005 | 8/32 | several |
| distill | +0.0125 | **-0.0007 +/- 0.0004** | **2/38** | **4** |

**Seat delta does not track solo marginal. It tracks how many times the six already seat the
attribute.** `distill` has the second-largest solo marginal on the honest pool and is the worst
possible seventh seat, losing 38 of 40, because the six carry it four times over (`stack_increments`
x2, `clean_calib`, `seat_clean_distill`) and one more distillate is redundant. The two attributes with
positive seat deltas are exactly the two least-represented ones, and both are worth about half a tick
of noise.

### What this explains, retroactively

Three closed results collapse into one mechanism. The diversity picks failed three times (routing
+0.0008 +/- 0.0033, coverage-greedy -0.0033, addition -0.0004 +/- 0.0007 losing 28/40) and the reading
at the time was that "least correlated with the six" is a proxy for WEAKEST. That was half right.
Least-correlated is a proxy for two things at once -- weakest, and least-represented -- and only the
second one pays. The rule failed because it could not tell them apart. D370's fixed-count reseating
going 0 for 57 is the same fact from the other end.

### What it licenses, and the correction to the earlier reading

D380's closing had argued that saturation licenses spreading attributes one per member. That
over-reached and was withdrawn: k=1 members sit ~10 ticks below k=5 members, and the diversity rule
that would seat a weak spread member has failed three times. But the correction runs the other way too.
"Keep stacking" now fails SPECIFICALLY for distill, which is the one attribute whose seat value against
this six is negative.

The licensed plan is therefore narrow: **keep building stacked members, vary the axes the attribute
table does not index -- teacher identity, label table, fold geometry -- and do not stack more distill
onto anything destined for these six seats.** `chainfour_v5blend` (knee-t9fzu) is the first arm written
under it: `chainfour` with one variable moved, the training label table, to a table seated zero times.

### And the honest size of all of it

The best axis on that table is +0.0005. The board gap is 0.011. Every lever in the table is roughly one
twentieth of the target and sits below the board's own 0.003 estimability floor. This finding is worth
having because it stops the card being spent on seats, not because it points at a seat worth taking.

---

## 2. The board gap is a modelling hole in eight columns and a gold-verified label wall in four

The metric is a macro mean over twelve columns, so a per-column lift divides by twelve and a column
sitting 0.10 below its neighbours carries a hundred times the recoverable mass of a seat swap. D382
asked where the headroom physically is, against the LABEL ceiling rather than against the pool.

Gold provenance, verified: the 469 adjudicated cells are exactly `v2_lone_dissent` intersect
`holdout909`, per column -- every dissent cell in the holdout was adjudicated, so the per-column dissent
base IS the gold n. Overall contradiction 0.8913, tear dissent 277/277, and contradicted cells are
v2=1 -> verdict 0 on 96.9%, i.e. the error process is directional over-call. Per-column "best" is
max-over-304 and selection-optimistic.

| column | gold n | ceiling | best AUC | gap best->ceiling | class |
|---|---|---|---|---|---|
| Medial Meniscus | 111 | 0.849 | 0.9135 | -0.065 | C |
| Lateral Meniscus | 89 | 0.872 | 0.8771 | -0.005 | C |
| Contusion | 120 | 0.861 | 0.8916 | -0.030 | C |
| ACL | 60 | 0.903 | 0.9009 | +0.002 | C |
| MCL | 17 | 0.949 | 0.8274 | **+0.121** | B |
| Lateral OA | 19 | 0.960 | 0.8395 | **+0.121** | B |
| Fracture | 0 | 0.967 | 0.8517 | **+0.115** | B |
| Effusion | 9 | 0.960 | 0.8662 | +0.093 | B |
| PF OA | 11 | 0.960 | 0.8733 | +0.086 | B |
| Synovitis | 6 | 0.961 | 0.8852 | +0.075 | B |
| Medial OA | 21 | 0.956 | 0.9002 | +0.055 | B |
| Baker's | 6 | 0.964 | 0.9317 | +0.032 | B |

Macro: ceiling 0.9301, best-of-304 0.8799, pool median 0.8470.

**Class C -- the label wall, four columns, gold-verified with real n.** Medial Meniscus (v2 contradicted
on 111 of 111 dissent cells), Lateral Meniscus (89/89), ACL (60/60), Contusion (100/120). The pool's
best members score AT or ABOVE the perfect-truth ceiling -- Medial Meniscus's best is **0.065 above it**.
A member cannot beat a perfect-truth ceiling by learning truth; it beats it by learning v2's errors,
which is what the top of the pool has evidently done. Further measured "gain" on these four columns is
more learned noise. Nothing to win here against these labels.

**Class B -- the modelling hole, eight columns.** Ceilings 0.949-0.967 with the best member +0.03 to
+0.12 below.

### The n<30 flag reads backwards here, and class B is stronger than it was written

Every class-B column carries a small gold n (0 to 21), which was flagged as too small to report. That
caution is right in general and wrong in this shape, because the gold n is a **numerator, not a sample
size**. MCL's 17 means "17 of the 909 holdout studies have a contested MCL cell": the rate is 17/909,
denominator 909, precisely estimated, and six-fold smaller than Medial Meniscus's 111/909.

Decomposing what the ceiling is built from,

    epsilon(col) = P(dissent) * P(v2 wrong | dissent) + P(unanimous) * eps_unanimous

the badly-estimated conditional is bounded in [0, 1] by construction, so for MCL the whole first term
is bounded in [0, 0.019] regardless of what those 17 cells say:

    epsilon(MCL) in [0.032, 0.051]   ->   ceiling(MCL) in [0.949, 0.968]

The reported 0.949 is already the worst case of that interval. **`+0.121` is therefore a lower bound on
the hole, and the bound is tight precisely BECAUSE the dissent stratum is small.** A small numerator is
what makes the interval narrow. (Contrast `discord_mask`'s sensitivity at 32 affirmed cells, where the
small n IS the denominator and nothing bounds the quantity -- that one stays unreadable.)

### The single point of failure

All eight class-B ceilings rest on `eps_unanimous = 0.033` multiplying the ~98% of cells nobody
contested. At 0.033 MCL's ceiling is 0.949 and its hole +0.121; at 0.10 the hole is +0.05; at 0.15 MCL
becomes class C and the finding inverts. **Its provenance is asked and outstanding (M389), and until it
is answered this entire table is conditional on that one constant.** Separately, Fracture has zero gold
cells, so its +0.115 rests on the unanimous leg alone and belongs one confidence notch below the other
seven however attractive its size.

### Two deflations, neither fatal

- "Best AUC" is max-over-304 and carries the project's selection-optimism constant, ~0.011, so every
  gap above is about 0.011 too large. Class membership does not move; sizes do, and Baker's +0.032
  probably vanishes.
- Offline macro does not transfer one-for-one. D372's split-level fit gives slope **+0.29 [+0.09,
  +0.49]** for board >= 0.935 (n=7). So closing MCL entirely is +0.0101 macro and roughly **+0.003
  board** -- one tick. Closing MCL, Fracture and Lateral OA together is +0.030 macro, about +0.009
  board, which IS the gap. One column is not enough. Three is.

### What follows

The four walled columns cap everyone, so if the hidden board labels share v2's error process the leader
pays the same ceiling and that wall is shared, not differential. The differential lives in the eight.
MCL, Fracture and Lateral OA are simultaneously the largest holes and the columns with the most
member-specific variance.

**And the fix is not loss-engineering.** The obvious move is to upweight those three in the objective,
and the project's standing rule forbids it -- keep the objective general, fix needs with balanced data.
The rule is right here for a concrete reason: all three are rare positives, so upweighting buys AUC on
them by trading calibration everywhere else, and macro AUC would hide the trade until the board did
not. The answer is data, and the outstanding ask (M389) is the coverage matrix over the external tables
already on disk plus the prevalence-versus-gap correlation that would tell us in one join whether these
are simply the rare columns.

---

## Population note

D382 ran on the honest pool at n=304 @holdout909; D381 on 134 packaged-honest members with candidates
from the 0%-exposure sub-pool n=125, both rosters walked. `greedy_halfsplit` and `diversity_seventh`
ran on 117, a single-shelf walk, and still may not be quoted until rebuilt.

One process note worth keeping. D381 reported `distill_stack_clean` as truncated at epoch 3 off a
`metrics.jsonl` that was still being written; the arm was live at epoch 4 of 12 with its log mid-epoch
at `step 550/582`. A live run and a dead one are indistinguishable on disk and `convergence_check`
cannot separate them -- only the process table can, and MSYS `ps` shows no argv, so the check is
`Get-CimInstance Win32_Process -Filter "Name='python.exe'"` filtered on `CommandLine`. The reading that
mattered, `S - S'` for the exposure pair, was not broken, only not yet takeable.
