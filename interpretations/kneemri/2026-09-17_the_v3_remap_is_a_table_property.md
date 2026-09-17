# The v3 re-map's gold gain is IN-SAMPLE: the rungs were fitted on the gold studies

*2026-09-17. Superseded its own first version, written 12:00Z the same day, which read the gain as real
and priced a roster rebuild on it. Corrected 12:18Z off
`python -m kneemri.evaluation.annotated_readings --device cpu --checkpoints recal_calib_s2026
calib_dense_hold_s2026 recal_dense_s2026 band_dense_hold_s2026`, which prints the defect itself.*

## The correction, first

`report_labels_v3.csv` replaces each of the reader's five confidence rungs with that rung's MEASURED
positive rate **on the 58 annotated studies** (`kneemri.evaluation.level_calibration`, knee-7oep):
0.08/0.28/0.68/0.82/0.94 becomes 0.0408/0.1356/0.3776/0.6879/0.9583.

The gold ruler **is those same 58 studies.** So a v3-trained member scored on gold is scored against a
table derived from the scoring set. `annotated_readings` says so in its own output and names the bead:

```
3 of them carry a constant FITTED on these cells, so their score here is agreement
between two rulers that share an origin, never a confirmation (knee-vrioa):
  recal_dense_s2026    --labels report_labels_v3.csv (derived, not the corpus report_labels_v2.csv)
```

This is the `raptor_solo` mechanism — a number read on studies the entry effectively trained on — on a
different axis. It voids the reading, not the member.

## What the numbers actually are

```
                      proxy ledger75    gold (58 annotated)
band_dense_hold_s2026       0.8877               0.8956      control, v2
recal_dense_s2026           0.8923               0.9128      v3 — gold IN-SAMPLE, void
calib_dense_hold_s2026      0.8949               0.9117      v2 + --calibrated-targets
recal_calib_s2026           0.8945               0.9157      v3 + --calibrated-targets

paired against recal_calib_s2026, resampled over studies:
  calib_dense_hold_s2026   -0.0040 [-0.0196, +0.0115]  INSIDE — no claim
  recal_dense_s2026        -0.0029 [-0.0180, +0.0110]  INSIDE — no claim
```

`--calibrated-targets` (`Supervision.MEASURED_AGREEMENT`) is fitted on the same 58 cells, so three of
the four rows above are in-sample on gold and only `band_dense_hold_s2026` is not.

**The only honest number this axis has is the proxy: `recal_dense` +0.0046 over `band_dense_hold`.**
That is below the 0.0062 paired floor and far below the 1.5% bar. Directional at best.

## What the first version got wrong, and how

The first version read +0.0172 gold against +0.0046 proxy and explained the 3.7x as structural — *"the
proxy's truth IS the report table, so it is biased against label-side movement by construction; read gold
first on this axis."* That inference is not available: the simpler explanation is that the gold gain is
in-sample and the proxy gain is the whole effect.

The mechanism of the error is worth more than the result. The reading was taken from `ledger75` and from
a saved `logs/gold_recal_paired.log`, and the in-sample warning is printed by `annotated_readings`
itself — the tool that was not run. **A saved log is a number without its caveats.** Re-run the reader
rather than grepping its output when the question is whether a number is admissible.

## The composition, settled on both rulers

`recal_calib_s2026` = v3 **with** `--calibrated-targets`, same corner. Pre-registered
`>= 0.8990 COMPOSES` / `0.8950-0.8990 PARTIAL` / `<= 0.8948 DOES NOT COMPOSE`.

- proxy: **0.8945 — DOES NOT COMPOSE**, below its own band, against parent `calib_dense_hold` 0.8949.
- gold: **−0.0040, interval spans zero — no claim**, and in-sample on both sides anyway.

The arm's own written mechanism predicted this before it ran: v3's fitted levels and
`LevelCalibration`'s pool-adjacent-violators fit are two answers to the SAME question — what positive
rate does this reader value mean — so the second has nothing left to correct.

`experiments/arms/recal_labelwise.sh` was that composition at labelwise pooling. It was on the card and
was **killed at epoch 1**, 20 minutes in, freeing ~1 h. Running it would have measured a collision
already measured at the neighbouring corner, against a gold ruler that cannot adjudicate it.

## What is NOT ruled out

The re-map is a monotone transform of the rungs, so every per-label rank AUC of the TABLE is unchanged
and only the gradient target moves. That remains true and remains a small lever by construction.

What this does not touch is the **per-finding** structure, which is not monotone and not uniform:
the reader's own error rates over the 58 studies run from 9.7% FP (Synovitis) to 69.6% (Effusion), and
the model beats its labels on 6 of 12 findings while losing 0.0692 on MCL. The board metric is an
unweighted mean over those 12 findings, so a per-column reliability weight changes the RELATIVE gradient
across findings in a way a monotone rung remap cannot. That is a different mechanism with a larger lever
arm — and it inherits this page's defect: any weight fitted on the 58 cells is unreadable on the 58
cells, so it needs a held-out adjudication set or it is this mistake again under a new name.

## The denominators

Proxy: 909 held-out studies, fold 0 of split_seed 2026, `ledger75` mixed cut. Gold: 58 annotated
studies, 696 cells, 34.5% positive, per-finding positives 9 to 35; all four members fitted
`--hold-annotations`, so none TRAINED on those cells — the contamination here is through the LABEL
TABLE, not through the split, which is why `--hold-annotations` did not prevent it.
