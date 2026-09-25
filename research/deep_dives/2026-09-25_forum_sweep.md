# Forum sweep — 2026-09-25

`python -m kneemri.kaggle.discussion_read --live --since 2026-09-21` at 08:20Z: 60 topics read, 11 posted
since 09-21. Three threads carried anything priceable. All claims are [U] (unverified, theirs) unless marked.

## 743148 — generate report text from images?

- Poster: LLM-extracted labels read 0.9083 macro AUC on the 58 gold studies; one fold scores 0.935 on the board.
- Reply (measured by them, unverified): image → report → labels only inserts a lossy text bottleneck.
  A linear probe on BGE-M3 report embeddings (PCA 128) scored 0.8423 on gold vs 0.8843 for LLM+regex
  extraction. Swapping the extraction LLM moved 0.010 across three Qwen models on 58 studies (noise).
  22 of 33 missed gold positives come from reports that never name the finding.
- Same reply: going from 24 to 32 cached slices (69% → 84% of slices retained) was "worth more than
  anything I did to the labels". **Ours is measured and opposite:** 24 → 36 read −0.0224 ([AXES:91](../../docs/AXES.md)),
  and 20 → 24 under labelwise reads +0.0005. Their feed is 2.5D windows, not our slot stack, so the
  claim does not transfer by construction.
- Third reply: EfficientNet-B0, 5 folds + TTA on ~0.88 labels reaches 0.94+ on the board. We ship
  above that (0.946).

## 742926 — improving a 0.940 ensemble

A reproduction of the public "Bend the Knee to the DINOv3" notebook (v27): 20 DINO members, five
DINOv3/A5 folds, RadImageNet heads, probability TTA. Reads 0.940. Asks, rather than tells. Nothing to price.

## 742327 — private shake-up

- A board-0.947 competitor: labels ~0.89 on gold, model 0.925 on gold. Trains on everything and trusts
  +0.002 board moves.
- Reply: "Top 50, everyone is now over 0.95"; predicts the top 10 over 0.96 once teams merge.
- Reply: much of bronze/silver is forks of two public notebooks, so shake-up risk sits there.

## What changes for us

Nothing to arm. The one mechanism claim (more slices) is already measured negative on our stack. The
"top 50 above 0.95" claim is consistent with our 09-23 read (#20 at 0.953): the gap to the medal band
is ~0.005-0.007 board, and to first ~0.011.
