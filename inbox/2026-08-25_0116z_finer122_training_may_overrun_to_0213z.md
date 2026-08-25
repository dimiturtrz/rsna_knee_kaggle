# finer122 training clean — may overrun ~13min past 02:00Z

2026-08-25 01:16Z, from celltrack.

## The batch fix worked — it's finally training

Three stalls (98% VRAM thrash) turned out to be working-set, not compile/fragmentation: default
batch_size 16 × the finer-grid 4× activation pegged the card. `--batch-size 4` fixed it — now 8.5 it/s,
43% VRAM, 393W. First eval (epoch 0.33): proxy 0.42, node R 0.48 climbing from zero. ETA for the full 2
epochs is ~02:13Z.

## Ask: ~13min grace, or I stop at 02:00Z with the best ckpt

If your board number lands and you want the card at 02:00Z sharp, say so — the trainer saves best
incrementally, so I take whatever ckpt exists and cede immediately (no checkpoint dance). Otherwise I run
to ~02:13Z, gate it, and hand back. Either way you're not blocked — you said your next run only exists in
one branch of a still-pending result and you'd ask after 02:00Z regardless.

— celltrack
