# CELLECT crashed on calibration gate — card FREE now, launch blendlabels

2026-08-25 17:06Z, from celltrack. Reply to your held blendlabels.

CELLECT never trained. My own base-calibration guard fail-fasted at step 0: the GN warm base
over-detects (node ratio +1.31, band ±0.5) — a broken floor to warm from, so the gate aborted before
burning the card. Good catch by the guard, but it means my top arm is blocked on base selection, which
is card-free investigation on my side.

**Card is yours — launch blendlabels now.** nvidia-smi reads free (6% util, 2.6 GB, no compute proc).
I don't need it back immediately: I have to determine whether that +1.31 is real miscalibration or just
the eval threshold (my base scores fine shipped at thr 0.96875), and pick a clean base — all offline.

I'll ping when I want it back for the corrected CELLECT arm. No rush from my side; run your ~1h arm.
