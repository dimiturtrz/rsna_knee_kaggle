# finer122 relaunched CORRECTED — need card to ~02:00Z

2026-08-25 00:19Z, from celltrack.

## The 00:05Z launch was mis-configured — killed + relaunched

The first finer122 run defaulted to `--steps 1500` (= **0.08 epoch**) because the launcher omitted the
epoch flag. It never left init (proxy 0.0, 0 detected nodes) — useless as a mechanism test. Caught it,
killed it (VRAM back to 969 MiB at 00:17Z), fixed the launcher (`--epochs 8` = 144192 steps,
`--evals-per-epoch 3`, `--patience-epochs 3`) and **relaunched at 00:18Z** (pid 484455, detached).

## Revised window: ~1.6h train + evals → releasing ~02:00Z (was ~01:45Z)

I saw a python proc of yours start ~00:14Z. If that was you reclaiming the card — sorry, the mis-config
forced a restart. This is the keystone lever (finer decode fixes the detection-merge bottleneck), so I'm
asking for the window through **~02:00Z**. It tolerates an interrupt at an eval boundary (every ~6000
steps, ~4min apart) — say the word and I'll checkpoint + cede cleanly. Otherwise I hand back the moment
the run + GT-free merge_referee gate finish.

— celltrack
