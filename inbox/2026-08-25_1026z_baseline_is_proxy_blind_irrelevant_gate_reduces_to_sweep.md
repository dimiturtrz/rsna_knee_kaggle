# The (1,4,4) baseline is proxy-blind-irrelevant — the go/no-go reduces to the sweep alone

2026-08-25 10:26Z, from biohub. Chased a wrong number, caught it, reframed. Recording so the submission
logic is clean when the sweep lands.

## The wrong turn

I started to want a "shipped (1,4,4) local faithful on TEST four" baseline — the number HOCT+finer must
beat before a submission is arguable. Went to fetch it. It's the wrong number.

## Why it's irrelevant

Bottleneck A (finer (1,2,2) decode) is **proxy-BLIND** — the oracle showed it separates cells <2µm that
(1,4,4) merges, but that recall gain is LB-visible only; it does NOT surface in the local faithful metric.
So a shipped-(1,4,4) faithful baseline can't see finer's merit. Computing it resolves nothing.

## The clean decomposition

The HOCT+finer stack = two INDEPENDENT named mechanisms, each justified on its own axis:

- **A — finer (1,2,2):** bottleneck A (detection-merge). Justified by the ORACLE (physical, <2µm
  separation), NOT by any local number. Proxy-blind by nature.
- **B — HOCT head:** bottleneck B (affinity confusor). Justified by the instrument (top1 0.012→0.214,
  non-inverting ruler, single-variable frozen-detector design). Locally validatable.

faithful 0.6877 (standard-finer) vs 0.6432 (HOCT-finer) therefore measures **HOCT's recall cost within
the finer grid** — NOT finer's merit (invisible locally). The sweep's only job: recover HOCT-finer recall
to ≥ standard-finer 0.6877 while top1 holds >> 0.012.

## Go/no-go (final form)

If the sweep produces an HOCT-finer config with faithful ≥ 0.6877 AND top1 >> 0.012 → the stack is TWO
mechanism-justified axes (oracle-A + instrument-B), neither leaning on a top-end proxy number the proxy
inverts on. That IS the submission thesis CLAUDE.md licenses. No (1,4,4) eval in the path.

If the sweep can't recover recall → HOCT's B-fix net-costs recall in this grid; hold the slot, and the
mechanism to try next is linker-side (disappearance-cost `--set`, or the two-pass tracklet ILP the paper
adds — less sensitive to per-edge score calibration than the flow linker).

Card still mine for the sweep. Scout crashed (autocompact thrash on a big read); did the recon inline.

— biohub
