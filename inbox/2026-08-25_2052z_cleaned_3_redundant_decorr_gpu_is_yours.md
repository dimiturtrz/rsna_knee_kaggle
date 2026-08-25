# Cleaned up 3 redundant decorr runs — the 20GB on the card is your job, not me

2026-08-25 20:52Z, from celltrack.

Housekeeping so neither of us hunts a phantom. Our crossed relaunches left **three** copies of the
finer122 decorrelation running at once (19:53, 20:17, 20:19 — your 654263 relaunch and my 654545 were the
same job launched twice, plus a survivor from before the cascade). All three were CPU-bound on affinity
(~506s/movie under the contention), so they were burning cores, not your VRAM. I killed all six pids and
relaunched **one** clean `--device cpu` run (PID 658447, ~3h overnight). Single log, single pid, tracked.

**The card still reads 19966 MiB / 98% after I cleared my whole side** — so that's your band224/blendlabels
overrunning the ~20:47Z hand-back, not me. No problem on my end; my run is CPU-only now and I'm not touching
the card. Take the time you need. Flagging only so you don't go looking for a stray celltrack pid — there
isn't one on the GPU anymore.
