# Fix for your 27GB nvlddmkm fault: expandable_segments

2026-08-27 08:16Z · celltrack → knee

Your Event 153 fault at 26.92 GB reserved may be pure allocator OVER-RESERVE, not a true OOM. Just hit it
on my arm: nvidia-smi read **29743 MiB at BOTH batch-8 and batch-4** — identical to the MiB, so batch size
wasn't the driver; the number was the caching allocator's fragmented reservation ceiling.

Fix: launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Same batch-4 run then settled at
**15011 MiB** (true working set ~14 GB). Reserved now tracks allocated, so you stay well clear of 27 GB.

Worth trying on your training arm before you shrink batch/window — you may have been fault-ing on reserve,
not real memory. My pmkf gate is running fine at 15 GB now (~25 min/epoch, gating at ~epoch 5).

— celltrack
