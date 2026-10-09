# The efficiency lane, priced — and closed by its own exchange rate

> **Dated snapshot (banner added 2026-10-09).** Its champion and rank numbers predate the public-checkpoint seat. Read today's pin from `comp/champion.json` and the open levers from `2026-10-09_levers_before_final_picks.md`. Its conclusions about OUR pool still bind.

2026-09-21. Zero slots. Two `serve_bench` runs, one `holdout_backfill`, one `ledger75`.

## Why it was opened

The main board pays **ten** prizes and we sit 76th of 4082, +0.008 from tenth. The efficiency board
pays **three** and we sit **17th** — 14 places from the money against 66. On rank-distance alone the
efficiency lane looked like the better ratio, and nobody had priced it.

Our efficiency row is `extffsolo_v1` (0.945, submitted 2026-09-17), read from the host's own notebook
output: `kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb` →
`full_leaderboard.csv`, 3976 rows, `EfficiencyRank, TeamName, PublicScore, DateSubmitted`.

## Three things that were believed and are false

**1. "The forward pass is invisible in our kernel wall-clock."** It was read off
`logs/kernellogs/v4solo/kneemri-v4solo.log:10` — `decode 6s for 1 read(s) at [[336]], elapsed 6s`.
That line proves nothing. `Submission.chunks` (`kneemri/kaggle/submission.py:272-296`) sets
`decode_started`, waits on the *prefetched* decode, logs, and only then `yield`s — the forward for that
chunk runs in the consumer, **after** the line is printed. `STUDY_CHUNK = 48` and the public test is
3 studies, so there is exactly one chunk and the log can never contain forward time. `elapsed == decode`
is structurally guaranteed, not measured.

Worse, it is unfixable from Kaggle: the served log is the *public-run* log, never the hidden rerun's.
Our real runtime split is unobservable from the platform and has to be measured here.

**2. "The cheap corner already lost head-to-head."** `cheapwide_v1` (224 px, 12 slices, board 0.928)
ranks below `extffsolo_v1` (336 px, board 0.945) on efficiency. But `cheapwide_v1` carries **ten
members** — ten forwards — against one. The comparison is confounded on member count, and says nothing
about a *single* cheap member, which has never been on that board.

**3. "A smaller trunk is the lever."** The trunk is already `dinov2-small`.

## The measurements

`python -m kneemri.inference.serve_bench --studies 48` on this 5090, the shipping member
`headlr_ship_s2026` (6 slots x 24 slices, 336 px):

    336 px   eager 3434.4 ms/pass   compiled 2645.2 ms   -23%   compile 28.4 s, payback 36 passes
    224 px   eager 1430.3 ms/pass   compiled 1104.6 ms   -23%

3434 -> 1430 is 0.42x against a predicted `(224/336)^2 = 0.444`. Forward cost is quadratic in read size
and therefore predictable from shape, which is what makes the rest of this arithmetic possible.

48 studies per pass is one kernel chunk, so 3434 ms is **71.5 ms/study** on a 5090.

The serving cap, priced without training — `holdout_backfill --served-slices 12` re-derives the holdout
through the *submission* path and writes a re-grouped record beside the checkpoint, so the 24-slice
record is untouched:

    headlr_ship_s2026                 0.9059 ledger75   24 slices
    headlr_ship_s2026_n12b0.2-0.8     0.9033 ledger75   12 slices
                                     -0.0026

## The exchange rate, which is what closes it

`Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400`, minimised. At Benchmark 0.5 and
maxAUC 0.957 the denominator is -0.457, so `d/dAUC = -2.188` per unit AUC and `d/dR = 3.086e-5` per
second. One unit of AUC is worth **70,900 s**, i.e. **0.001 AUC = 70.9 s**. Call that a tick.

Every cheapening we can price, at ~1300 test studies (estimate, `research/deep_dives/2026-09-04_qdu-foreign-
architecture-census.md:163`) and a ~6x T4 penalty over this 5090 (estimate, unmeasured):

| change | runtime saved | ticks gained | AUC cost | ticks lost | net |
|---|---|---|---|---|---|
| 336 -> 224 px | ~330 s | +4.6 | -0.024 (band224 0.8801 vs 0.9045) | -24 | **-19** |
| dinov2-small -> resnet50d | — | — | -0.013 (0.9078 vs 0.8948) | -13 | **-13** |
| 24 -> 12 slices | ~280 s | +3.9 | -0.0026 (measured above) | -2.6 | **+1.3** |
| torch.compile | ~128 s - compile | ~+1 | 0 | 0 | **~0** |

Only the slice cap is positive, and it is +1.3 ticks with **both** terms inside their uncertainty: the
-0.0026 is under the 0.0053 paired floor, and the +3.9 rests on two estimates neither of which we can
measure. Directional. Not a result.

**The bound that ends the search.** If our entire runtime went to zero we would gain `R / 70.9` ticks.
At an estimated R ~ 1140 s that is **~16 ticks**. Teams ranked above us are up to **12 ticks** better on
AUC alone (rank 3 tomoon33 at 0.954 against our 0.945 is 9). So a podium on efficiency needs an
AUC-preserving cut of **600 s or more** — over half the whole budget — and the best AUC-preserving knob
we own yields 280 s at a 2.6-tick price.

The formula weights AUC far too heavily for a cheap model to win it. The efficiency prize is won by the
**accurate** entries that are also fast, which is why rank 1 (Scott Willis, 0.957) is also the top of the
main board.

## What is still live

`Ibrahem-A163` ranks **16 at 0.936**, nine ticks below us and still ahead, which bounds them at
**>=638 s faster than our kernel**. That is roughly our entire forward pass. Something in our serving
path is slow and we do not know what, because the split is unobservable from Kaggle and only the forward
half has been measured here. The decode half (`core.data.corpus.decode_bench`: 0.12 s/study with 4
processes on 16 fast cores; Kaggle has 4 slow ones) has never been measured at Kaggle's core count.

That is a real unexplained lever, but it is a lever on a prize whose arithmetic we have just shown we
cannot reach. It is recorded, not queued.

## Consequence for the program

The efficiency lane joins the closed axes in `knee-fbo5e`. The deliverable stays a single solo
checkpoint judged on the main board, and the +0.008 to tenth place is not in this lane.
