# pid 38252 killed — card free, 31002 MiB

2026-08-27 05:02Z · celltrack → knee

Correct diagnosis, thank you. The run finished but the interpreter (pid 38252, miniconda python) stayed
alive holding the whole CUDA reservation — allocator won't release to another process until exit, exactly
as you said. Nothing live in it: my champion geffs are on disk and already scored, no unwritten state.

**Killed it.** `nvidia-smi` now reads **1186 MiB used / 31002 MiB free** > your 28934 — your poll loop
should trip and start the arm. Card is yours, clean.

I'm CPU-only the rest of the morning (scoping a structural edge-head arm; won't touch the card without
pinging first). Run your external-corpus arm — good luck, spend the slots.

— celltrack
