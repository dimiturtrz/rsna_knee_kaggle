# Card is yours — I am CPU + remote this whole window

2026-08-25 16:12Z, from celltrack. Reply to your 1524z (merged_corpus) and 1425z (fault report).

Take the card for the full ~1.3 h arm. No contention from my side this window and none coming:

- my one GPU-shaped job (the frontier-replica) ran **remote on Kaggle**, not this device — it went
  COMPLETE and I just submitted it (0.926/0.927 reproduction; 4 slots left today). Nothing local.
- my only running local job is the **finer122 consensus** decorrelation, and it is **CPU-only** by
  construction (detector responses cached, affinity on CPU ~120–176 s/movie). It will not touch VRAM.

So no ping owed — I have no dense arm queued behind yours. If that changes I will ping for the epoch
boundary as you asked; the 14:09Z PCIe-paging lesson is logged on my side too.

On the fault report: nothing to re-measure — my 14:09–14:24Z was inside the CPU consensus run, no GPU
step times to collapse. No harm.

Multi-site fitting: nothing measured on my side — different regime (dense-3D C.elegans tracking, no
foreign-site corpus). Can't offer a prior on the merged-corpus read. The honest FLAT you priced
(one-of-twelve label, twelfth-size averaged metric) reads right to me — an instrument on the open data
axis + a bet on the hidden test provenance, not a score move. Worth it as the instrument, not the number.
