# The shipping corner's learning rate was inherited from a 12-epoch sweep and never re-asked

2026-09-17. Bead knee-e9f83. No card spent to produce this — it is a read of 406 arm scripts joined to
the holdout records already on disk.

## What was looked for

The project ships `labelwise_depth24_drop40_long_s2026` at 0.9053 ledger75. Eighteen of the 406 arms
in `experiments/arms/` sit at that corner — `--epochs 24 --slice-pooling labelwise --slot-dropout 0.40`.
The question was which flags have been varied there and which have not.

Varied across those eighteen: labels (four schemes), slices (12/20/24), band-low (0.1/0.2), band-high
(0.8/0.9), batch (4/10), image-size (224/336), vram-gb (22/24/28), scheme (four values).

Fixed across all eighteen: `--backbone-lr 4e-5`.

Widening to the whole corpus, 334 arms name the flag and 286 of them use 4e-5. Every arm carrying any
other value — 1e-5, 2e-5, 3e-5, 8e-6, 8.2e-5, 9.8e-5, 1e-4, 1.26e-4, 1.2e-4, 1.5e-4, 2e-4, 4e-4, 5e-4,
1.2e-3 — runs **12 epochs**. At 24 epochs the corpus holds 36 arms at 4e-5 against one at 1e-5, and that
one is `warm_hislabels`, a warm start whose low rate is by design rather than by sweep.

So the rate was chosen on a 12-epoch schedule and carried into the 24-epoch labelwise corner for 41 arms
without being varied once.

## Whether it was ever the measured optimum

Inheriting a constant is only a defect if the constant is wrong, so the next question is what the
12-epoch sweep actually said. The arms are not a sweep — they are mostly foreign-recipe replications
carrying their own published rates, confounded on backbone, pooling, slices and band all at once. The
only readings that mean anything are pairs whose every other flag is byte-identical.

Sweeping all 333 arms that carry both `--backbone-lr` and `--out` for such pairs gives sixteen. The
scored ones, ledger75 over the 909-study holdout:

| pair | low rate | high rate | delta |
|---|---|---|---|
| `llrd224_flat` → `llrd224` | 4e-5 **0.8694** | 8.2e-5 **0.8737** | **+0.0043** |
| `band_dense_llrd` → `band_deep_matched` | 4e-5 **0.8765** | 9.8e-5 **0.8781** | **+0.0016** |
| `swin_calib_dense` → `swin_wholetop` | 4e-5 0.8479 | 1e-4 0.8509 | +0.0030 |
| `swin_calib_dense` → `swin_supervised` | 4e-5 0.8479 | 1e-4 0.8537 | +0.0058 |
| `ortho_dense` → `ortho_fresh_lowlr` | 4e-5 0.8748 | 1e-5 0.8526 | −0.0222 |
| `ortho_pool_read` → `ortho_slow_trunk` | 4e-5 0.8760 | 1e-5 0.8540 | −0.0220 |
| `warm_calib_dense` → `warm_calib_lowlr` | 4e-5 0.8916 | 1e-5 0.8910 | −0.0006 |
| `cnx_own_recipe` → `cnx_top_rate` | 4e-4 0.8451 | 1.2e-3 0.8042 | −0.0409 |
| `mricore_corpus` → `mricore_rate` | 4e-5 0.8658 | 4e-4 0.8034 | −0.0624 |

The first two are our own DINOv2 whole-stack recipe, and both diffs were read directly to confirm they
touch nothing but the rate (and the name and `--out` that follow it).

Read as a curve: 1e-5 is far down on a fresh trunk (−0.022, twice) and flat on a warm start (−0.0006,
where the trunk is already placed and the rate has little left to do); 4e-5 and ~1e-4 are within a few
thousandths of each other with **every** pair tilting upward; 4e-4 and above collapse. The optimum is a
plateau somewhere between 4e-5 and ~1.5e-4, and 4e-5 sits on its low shoulder.

**Four for four on sign** going up about 2×, median about +0.0037. Each of those is individually under
the 0.0062 paired floor and none is a finding on its own. What is a finding is that the curve is
asymmetric around 4e-5 and tilted, and that 4e-5 was never the measured top of anything.

## Why 24 epochs should want more rate than 12, not less

Every datum above is from a 12-epoch arm with no slot dropout. The shipping corner changes two things
that both push the same way:

- a one-cycle schedule doubled 12 → 24 at a fixed peak rate;
- `--slot-dropout 0.40`, a large added regulariser that was not present when the rate was chosen.

Both argue the corner is further from overfitting than the sweep's arms were, and the training loss says
so directly: the shipping arm finishes 24 epochs at train loss **0.5600**, while `calib_dense_hold_s2026`
— same rate, half the budget — finishes at **0.5130**. The arm with twice the schedule fits *less*. That
is the signature of a run that is regularisation-bound rather than capacity-bound, and it points in the
same direction as all four pairs.

## What is not claimed

The mechanism argument is an argument, not a measurement: nothing in this repo has varied the rate at
24 epochs, which is the whole point of the bead. The 12-epoch tilt may simply not survive the doubled
schedule, and that outcome is pre-registered as its own band rather than as a null.

Nor is this a 1.5% lever. The honest expectation is a few thousandths, priced at 2.6 h of card against
`P(clears 0.0062) ~ 0.20`. It earns its slot because the control already exists at 0.9053 — no matched
baseline has to be paid for — and because an inherited constant on the axis that sets every other
constant's meaning is worth one arm.

## The general shape of the error

This is the same defect as the ledger's twelve populations found the same afternoon, one level up. There
the ranking compared numbers read on different study sets and said nothing about it; here the corner
carries a constant fitted under a different schedule and says nothing about that either. In both cases
nothing was broken and every check was green — the missing thing was a record of what the number is a
statement *about*.
