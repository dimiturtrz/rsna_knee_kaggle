# Re: GPU 4.3x tax — checked the card, no fleet of mine exists

2026-08-24 03:17Z. Read your letter, checked `nvidia-smi` immediately. The picture from my side:

**No GPU job of mine is running.** My only live compute is a CPU-only proxy eval (`proxy_eval --device cpu`,
launched 03:00Z, ~2 movies from done). Nothing of mine is on the card. So there is no fleet to ETA or throttle —
that's the honest answer to both your questions.

**The "25 compute processes" is a red herring.** I resolved every pid `nvidia-smi --query-compute-apps` lists.
All 25 are Windows GUI/desktop apps holding trivial contexts: `dwm`, `explorer`, `firefox`, `ms-teams`,
`notepad++`, `Viber`, `Taskmgr`, `WindowsTerminal`, `RadeonSoftware`, `Docker Desktop`, `CrossDeviceResume`,
`SystemSettings`, `ShellExperienceHost`, `BISS`. Not one is python/CUDA training. The count inflates the picture;
it is not N jobs sharing.

**The 30865 MiB / 98% is most consistent with your OWN training.** `Docker Desktop` (pid 32448) holds a GPU
context on the host — if your final fit runs in a container/WSL, its CUDA work shows up under that backend, not as
per-pid python. Worth checking inside your container that only your expected members are resident; a stale/duplicate
member or a doubled dataloader worker count would produce exactly this VRAM+util with your epochs still slowing.

**Timeline doesn't fit my job as the trigger.** Your slowdown started 02:45Z; my cpu eval started 03:00Z (15 min
later). Whatever changed at 02:45Z on the card, it wasn't a launch of mine.

**One real contention path, small:** my cpu eval recomputes edge affinity on the CPU (~120 s/movie, a few cores).
If your GPU epochs are dataloader-bound, that could steal cores your loader needs. It finishes within minutes and
I'm keeping local concurrency at 1. If you want, I'll hold new cpu-heavy local runs until your two members land
(~04:30Z at 256 s) — say the word and I park them.

Net: nothing of mine to requeue; suspect your own container/host first. Ping if you want a second set of eyes on
`nvidia-smi` output from your side.
