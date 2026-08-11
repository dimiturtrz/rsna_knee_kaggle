# experiments — the arms, in git, runnable by shell

Every number this project reports came out of one of these. They lived in `/tmp` until 2026-08-11,
which meant the definition of each measurement was unversioned, unreviewable, and one reboot from
gone — the same failure the no-notebooks rule exists to prevent. An arm is code. It belongs here.

```bash
bash experiments/arms/compose_base.sh            # one arm, foreground
nohup bash experiments/arms/whole_stack.sh &     # one arm, detached
```

Each arm sources `lib.sh`, which resolves the data root through `core.paths` (never a hardcoded
`D:/data`), fails on a missing input **before** spending GPU, and tees full output to
`<processed>/experiments/<arm>.full.log`. Logs are data and stay out of the repo; arms are code and
stay in it.

An arm file carries its hypothesis, its control number, and its result in the header. Reading the
directory is meant to be the same as reading the ledger.

## Two rules encoded in `lib.sh` rather than remembered

- **`wait_for` keys on a completion marker.** A gate that watches an output path fires when the
  producer *starts*. That once queued an arm against a 59.7 GB cache still mostly zeros; it would not
  have crashed, it would have produced a plausible wrong number.
- **`run_train` tees before it greps.** Arms used to pipe straight into `grep`, so any traceback whose
  wording missed the filter vanished and the run looked merely empty.

## Ledger — fold 0, seed 2026, 24 epochs, one-cycle, weight floor, ranking on

Control for the small arms is `rank24_s2026` = **0.8272**; for base arms, `rankbase24_s2026` = **0.8283**.
The noise floor on this ruler is **0.0062**, and this project's bar for a *result* is ~1.5% absolute —
so everything below is **directional**, and only the cumulative total clears the bar.

| arm | change | score | Δ | verdict |
|---|---|---|---|---|
| `rate_isolate` | encoder rate 8e-6 → 4e-5, depth 6 | 0.8323 | +0.0051 | kept, single-variable |
| — | same rate at depth 12 | 0.8316 | +0.0044 | depth is not a lever |
| `slice_count_maxpool` Q | max slice pooling, 12 slices | 0.8310 | +0.0038 | kept |
| `slice_count_maxpool` R | max pooling, 20 slices | 0.8309 | −0.0001 vs Q | **refuted** |
| `severity_flat` | three YES levels collapsed to one | 0.8228 | −0.0044 | gradation is live signal |
| `report_supervision` | report-embedding term @0.3 | 0.8219 | −0.0053 | **refuted** |
| `report_supervision` | report-embedding term @1.0 | 0.8209 | −0.0063 | refuted, monotone |
| `compose_base` | rate + pooling on base | running | — | shipping candidate |
| `whole_stack` | all 12 slices per step | queued | — | leaves the design family |
| `resolution` | 336 → 448 px | queued behind its cache | — | per-label prediction registered |

`compose_small` was **dropped from the queue unrun**: an hour of GPU to confirm that two sub-bar
levers add cannot change any decision.

## Why the arms above mostly could not have mattered

Every row that moved less than 0.006 is a knob *inside* one design, and that design uses about a tenth
of the data it is handed — 18 of ~180 acquired slices per step, 12 cached of a median 30 per series,
17.4% of series never routed to a slot at all, 336 px for findings a few voxels wide. We are at 0.888
on the leaderboard against a 0.942 frontier, and a knob inside a family returns ~0.005 by
construction. `whole_stack` and `resolution` are the first two arms that leave it. See `knee-jk3`.

Interpretation of the whole ledger, including what each refutation rules out, lives in
`interpretations/kneemri/`. The build log is git history.
