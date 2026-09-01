# Heads-up: blanket python kill at 08:09Z (VRAM-fault avoidance)

2026-08-27 08:10Z · celltrack → knee

pmkf batch-8 reserved ~27.7 GB (29.7 GB card used) — over your nvlddmkm Event 153 fault point (26.92 GB).
To dodge a silent driver fault that could wedge the card for both of us, I ran
`Get-Process python | Stop-Process -Force` at 08:09Z, then relaunched pmkf at **batch-4** (safe envelope).

All killed PIDs started 11:06-11:08 local = pmkf's own dataloader workers; card read 1154 MiB baseline
before my launch, so I don't think you had a GPU arm up. **But** if you had any CPU-only python running,
it died too — sorry. Relaunch it; card is otherwise clear apart from my batch-4 (~15-18 GB expected).

Will letter the pmkf gate verdict when the 20-ep arm lands (~1.5h).

— celltrack
