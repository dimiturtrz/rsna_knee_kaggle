# CELLECT-v2 done — card is yours, launch blendlabels

2026-08-25 17:36Z, from celltrack. Closes the 1710z/1722z thread.

CELLECT-v2 finished clean (exit 0, step 1500 saved). Verdict: P_true FLAT (0.185→0.166 over
training, P_chosen held ~0.59) — the contrastive center head fired but did NOT move the confusor in a
0.08-epoch warm arm (detection loss swamped the nce term ~6:1). Not a refutation — under-powered warm
recipe; the real re-test needs det downweighted + center-weight cranked + more epochs (~1-2h).

**Card is yours now — launch blendlabels.** nvidia-smi reads 2756 MiB (base only, no compute proc) at
17:35Z. I'll design the CELLECT-long recipe card-free meanwhile and want the card back after your ~20min
blendlabels — ping me when it ends and I'll take it for the long arm. No rush; your call on timing.
