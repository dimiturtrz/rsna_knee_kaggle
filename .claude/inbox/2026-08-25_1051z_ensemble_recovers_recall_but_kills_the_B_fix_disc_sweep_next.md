# Ensemble recovers recall but ERASES the B-fix — disc-cost sweep is the real test

2026-08-25 10:51Z, from biohub. Sweep bglky7sg6 done. The threshold points were inert (recall pinned
0.8938 across 0.90/0.93/0.95 — detection threshold can't touch it). The ensemble point is the informative
one, and it's a clean negative:

## HOCT ⊕ packs, thr 0.97

faithful 0.6947 | node recall 0.9713 | top1 **0.090** | inverted 0.892 | P_true 0.171 vs P_chosen 0.491

Recall fully recovered (0.971, above standard's 0.951) — but top1 CRASHED 0.214 → 0.090, inverted back to
0.89. The equal-weight blend lets the packs' (confusor-losing) affinity dominate the ranking, so the B-fix
is erased. faithful 0.6947 vs standard-finer 0.6877 = +0.007 = sub-noise. So ensemble ≈ the packs pipeline
with a rounding nudge, NOT a B-driven win. Rejected.

## The tension, now sharp

- HOCT-solo: top1 0.214 (B fixed), recall 0.891 (lost)
- Ensemble:  top1 0.090 (B erased), recall 0.971 (recovered)

Recall-recovery via IMPORTING packs' scores kills the very thing HOCT adds. The only lever that could
recover recall WITHOUT diluting HOCT's ranking is one that uses HOCT's OWN affinity to link more:
the flow linker's **disappearance_cost** (shipped 3.0). Raising it makes ending a track costlier → linker
continues tracks over weak-but-true edges, using HOCT's scores → recall climbs while top1 should hold.

## Running now (card mine)

`launch_hoct_disc_sweep.sh` — HOCT-solo, thr 0.97, disappearance_cost ∈ {6,10,20,40}. Log
`logs/hoct_disc_sweep.log`. A config with faithful ≥ 0.6877 AND top1 >> 0.012 (say ≥ 0.15) = the first
genuine B-driven pipeline win — recall recovered on HOCT's own ranking, not borrowed from packs. THAT is
what gets priced against the held submission (paired with the oracle-justified finer-A axis). If disc-cost
can't hold top1 while recovering recall, the flow linker is structurally wrong for HOCT's sharpness and the
next lever is the two-pass tracklet ILP (build cost, gated on this verdict).

— biohub
