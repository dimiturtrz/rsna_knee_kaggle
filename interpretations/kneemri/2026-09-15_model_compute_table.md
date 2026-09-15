# Model compute table: every trunk with a trained seat, priced in FLOPs, VRAM, wall-clock and ledger75

Asked: "how much FLOPs per sample, how many params, how many samples per batch", to decide whether a smaller
trunk can run the grand recipe (knee-feb62) faster. No card used; everything below is from configs on CPU,
run logs and holdout records already on disk.

## How each column was measured

- **Params total / trained**:
  - HF models: `AutoConfig.from_pretrained(local_files_only=True)` then `AutoModel.from_config`.
  - timm models: `create_model(pretrained=False, num_classes=0)`.
  - Everything was instantiated on CPU with no downloads.
  - "Trained" counts the parameters in the **last 6 blocks**, which is `unfreeze_last 6`, the setting every seat below uses. The head is excluded because it is small.
  - resnet50d ran with `unfreeze 16`, so all of its blocks are trained.
- **GFLOPs per slice**:
  - Counted with `torch.utils.flop_counter.FlopCounterMode` on CPU: one slice, batch 1, trunk forward only, no head.
  - FlopCounterMode counts one FLOP per multiply, the same convention as timm's "GMACs" tables. Attention is included.
  - CoAtNet-rmlp-2 and Swin-B only accept 384 px (the `rmlp`/window shape, `foreign_encoders.py:217` comment), so they are counted at 384.
- **GFLOPs per study** = GFLOPs per slice × slices.
  - This is the **serve** cost.
  - Training with slice dropout (`drop40`) forwards fewer slices, which is visible in img/s, not in this column.
- **Train GFLOPs per study** = F × (1 + 2·k/L), where k is trained blocks and L is total blocks. Frozen early blocks need no backward pass.
  - The factor is 2.0× for a 12-block ViT, 1.5× for a 24-block ViT and 3× for resnet50d with every block trained.
  - This is **analytic**, not counted.
- **Batch / accumulation, s/epoch, img/s, peak memory**: parsed from `$MODELS/experiments/<name>.full.log`.
  - s/epoch and img/s are medians over the run's epochs.
  - Peak memory is the last epoch's `peak memory A GB reserved, B GB allocated`.
- **Max batch at 60 %**: the owner's VRAM cap is 50–70 %, and 60 % of 32 GB is 19.2 GB.
  - Estimate = floor(19.2 / (allocated / batch)). **UNVERIFIED.**
  - It ignores fixed weight and optimizer memory, which is < 1.5 GB even for ViT-L at k=6.
  - It assumes the reserved−allocated gap is closed, for example with `expandable_segments`.
  - **Reserved** memory (28.8 GB on the dense seat) already exceeds the cap at the batch that ran.
- **Score**: `python -m kneemri.evaluation.ledger75 --holdouts $MODELS`, run over all 419 records (909 rows each).
  - Column shows `LEDGER75 (midpoint)`.
  - The best honest run per trunk is shown.
- Every run has 20 slices at 336 px and a 130 mm crop unless the row says otherwise.

## The table (RTX 5090 32 GB)

| # | trunk / example run | params total / trained | input (slices × px) | tokens/slice | GFLOPs slice | GFLOPs study (train≈) | batch×acc | peak alloc / reserved GB | max batch @60% | img/s | s/epoch → run | ledger75 (mid) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dinov2-small · `labelwise_dense_drop40_long` | 22.1M / 10.7M | 20 × 336 | 577 | 24.8 | 495 (990) | 12×1 | 16.4 / 28.8 | 14 | 2670 | 185 s → 24 ep 74 min | **0.9048** (0.8672) |
| 2 | dinov2-small · `labelwise_depth24_drop40_long` | 22.1M / 10.7M | 24 × 336 | 577 | 24.8 | 594 (1188) | 10×1 | 15.9 / 27.6 | 12 | 2629 | 221 s → 24 ep 88 min | 0.9053 (0.8679) |
| 3 | dinov2-small · `labelwise_ext_drop40_long` | 22.1M / 10.7M | 20 × 336 | 577 | 24.8 | 495 (990) | 4×3 | 7.5 / 16.4 | 10 | 3582 | 205 s → 24 ep 82 min | 0.9045 (0.8672) |
| 4 | dinov2-small · `final_ws_cf` / `stack_increments_ff` / `through_dense_cf` (EMA, fold −1) | 22.1M / 10.7M | 20 × 336 | 577 | 24.8 | 495 (990) | 12×1 | 25.0 / 27.0 | 9 | 1126–1273 | 415–469 s → 12 ep 83–94 min | dark (fold −1, no holdout) |
| 5 | dinov2-small · `calib_span_ff` (fold −1) | 22.1M / 10.7M | span × 336 | 577 | 24.8 | — | 12×1 | 15.1 / 16.3 | 15 | 1246 | 257 s → 12 ep 51 min | dark |
| 6 | dinov2-small · `base224_drop40_long` | 22.1M / 10.7M | 12 × 224 | 257 | 11.0 | 132 (265) | 12×1 | 4.9 / 11.5 | 46 | 5181 | **45 s** → 24 ep 18 min | 0.8865 (0.8552) |
| 7 | dinov2-small · `crop160_span` | 22.1M / 10.7M | 12 × 420 | 901 | ≈41 (analytic, token-scaled) | ≈490 | 12×1 | 23.3 / 25.1 | 9 | 662 | 413 s → 12 ep 83 min | 0.8873 (0.8564) |
| 8 | dinov2-base · `base_clean` | 86.6M / 42.5M | 20 × 336 | 577 | 98.5 | 1971 (3942) | 6×2 | 26.0 / 27.5 | 4 | 499 | 927 s → 12 ep 185 min | 0.8887 (0.8570) |
| 9 | dinov2-base · `base224` | 86.6M / 42.5M | 12 × 224 | 257 | 43.9 | 527 (1053) | 12×1 | 14.0 / 15.6 | 16 | 1168 | 239 s → 12 ep 48 min | 0.8795 (0.8509) |
| 10 | dinov3-small · `latent_register_dense` | 21.6M / 10.6M | 20 × 336 | 446 | 19.2 | 384 (768) | 12×1 | 16.1 / 25.4 | 14 | 2246 | 207 s → 12 ep 41 min | 0.8619 (0.8385) |
| 11 | dinov3-small+ (no run found) | 28.7M / 14.2M | 20 × 336 | 446 | 25.5 | 510 | — | — | — | — | — | — |
| 12 | dinov3-base + LoRA r8 (one run, no epoch finished) | 85.7M / LoRA only | 20 × 336 | 446 | 76.3 | 1526 | 6×1 | — | — | 198 | — | — |
| 13 | dinov3-large · `v3large_stack` | 303.1M / 75.6M | 20 × 336 | 446 | 270.1 | 5401 (8100) | 1×12 | 6.3 / 6.5 | 3 | 305 | 1595 s → 12 ep 5.3 h | 0.8525 (0.8277) |
| 14 | orthofoundation-L (dinov3-L) · `ortho_dense` | 303.1M / 75.6M | 12 × 224 | 201 | 121.7 | 1460 (2190) | 12×1 | 15.7 / 17.3 | 14 | 720 | 402 s → 12 ep 80 min | 0.8748 (0.8452) |
| 15 | convnextv2-base · `cnx_warm_anneal` | 87.7M / 31.8M | 20 × 336 | 100 (10²) | 68.5 | 1371 (≈2370) | 1×6 | 6.1 / 7.5 | 3 | 662 | 770 s → 6 ep 77 min | 0.8374 (0.8142) |
| 16 | swin-base-384 · `raptor_feat_fit` | 86.9M / 37.9M | 20 × 384 | 144 (12²) | 88.9 | 1778 (≈3330) | 1×6 | 4.9 / 5.7 | 3 | 516 | 934 s → 12 ep 187 min | 0.8578 (0.8334) |
| 17 | coatnet-rmlp-2-384 · `coatnet_hybrid` | 72.9M / 35.5M | 20 × 384 | 144 (12²) | 85.7 | 1713 (≈3380) | 1×6 | 10.6 / 13.4 | 1 | 403 | 1215 s → 12 ep 4.1 h | 0.8467 (0.8259) |
| 18 | coatnet-rmlp-2-384 · `raptor_v4_swa` (his trainer) | 72.9M / all | ≈16–20 × 384 | 144 | 85.7 | ≈1700 | UNVERIFIED | UNVERIFIED | — | — | not in our logs | 0.9425 **VOID** (98.6 % in-sample) |
| 19 | resnet50d · `resnet_224_short` (all blocks) | 23.5M / 23.5M | 20 × 224 | 49 (7²) | 8.7 | 173 (519) | 4×2 | 7.6 / 24.3 | 10 | 2413 | 185 s → 12 ep 37 min | 0.8842 (0.8527) |
| 20 | vjepa2-L · `vjepa_warm` | ≈300M (not instantiated) | UNVERIFIED | — | — | — | — | — | — | — | not parsed | 0.8673 (0.8337) |
| 21 | rad-dino · `raddino_normfix` | 86.6M / 42.5M | 20 × 336 | 577 | 98.5 | 1971 (3942) | 6×1 | 25.5 / 27.1 | 4 | 507 | 913 s → 12 ep 183 min | 0.8396 (0.8155) |
| 22 | radiodino-s16 · `radiodino_s16` | 21.7M / 10.6M | 12 × 224 | 197 | 8.5 | 102 (204) | 12×1 | 5.0 / 6.3 | 46 | 4533 | 61 s → 12 ep 12 min | 0.8390 (0.8108) |
| 23 | radiodino-b16 · `radiodino_corpus` | 86.0M / 42.5M | 20 × 336 | 442 | 75.6 | 1512 (3024) | 2×3 | 6.7 / 8.0 | 5 | 731 | 636 s → 12 ep 127 min | 0.8499 (0.8214) |
| 24 | radiodino-s8 · `radiodino_s8` | 21.7M / 10.6M | 12 × 224 | 785 | 33.5 | 401 (802) | 12×1 | 18.8 / 24.8 | 12 | 934 | 297 s → 12 ep 59 min | 0.8196 (0.7971) |
| 25 | mricore · `mricore_corpus` / medsam ViT-B · `medsam_dense` | not instantiated | 20 × 336 | — | — | — | 2×3 | 6.8 / 7.7 · 9.2 / 11.5 | 5 · 4 | 730 · 506 | 636 · 920 s | 0.8658 (0.8390) · 0.8363 (0.8137) |

Tails not in the table:
- vit-base-in21k · `vit_in21k`: 86.4M, 75.6 GFLOPs/slice at 336, 1039 s/epoch, 0.8037.
- radimagenet densenet121 · `radnet_pinned`: 7.0M, 12.7 GFLOPs/slice at 336, 197 s/epoch, 0.7704.
- dinov2-large at 336: 304.4M total, 75.6M in the last 6 blocks, 349.2 GFLOPs/slice. No ledgered run.

## Caveats that change a reading

- **s/epoch is not FLOPs.**
  - Row 1 (495 GFLOPs/study) runs 185 s/epoch, while row 4, which has the same trunk and input, runs 415–469 s. EMA and fold −1 (≈25 % more studies) plus a different slice drop explain the gap.
  - The foreign trunks run batch 1 × acc 6, so their img/s mostly measures loader and Python overhead, not the card. A CNN/Swin 336 run would be faster at a real batch.
  - The throughput numbers above are what those recipes did. They are not a ceiling.
- **The 0.90+ small-224 rows are not honest comparisons, and are left out.**
  - `oof_gen2_fold3` scores 0.9028 at 67 s/epoch and `recal_holdannot` scores 0.9004.
  - The `oof_*` fold members read between 0.8873 and 0.9028 on the same 909 rows depending on fold, so their training folds overlap the ledger holdout.
  - `recal_holdannot` trains on a different annotation split.
  - The honest 224 reading is row 6, 0.8865.
- **Foreign dinov3 rows are also left out of trunk selection.**
  - `dinov3f0_solo` 0.9021 and `foreign_dinov3_matched` 0.8964 come from a foreign trainer, with trunk size and overlap not established here.
- **Raptor 0.9425 is in-sample.** `foreign_overlap` measured 98.6 %, and the board row was voided on 2026-08-29. It must not be read as a trunk ranking.
- `ortho*` rows are licence-gated by their fastMRI provenance.
- **Reserved vs allocated memory.** Every 336 dinov2-small seat at batch 12 already reserves 27–29 GB, above the 60 % line. Either the batch drops to about 9–14, or the allocator gap has to be closed. The max-batch column assumes the latter and is UNVERIFIED.

## Recommendation: cheapest trunk within 0.01 of the dense drop40 seat (0.9048 ledger75)

1. **No smaller or cheaper trunk qualifies. dinov2-small at 336 is already the cheapest trunk within 0.01.**
   - Honest runs at ≥ 0.8948 are all dinov2-small at 336 px with 20–24 slices:
     - roinative 0.9018
     - warm_calib 0.9000
     - chainfour 0.8982
     - clean_calib 0.8978
   - The next best trunks sit about 0.016 below and are no cheaper:
     - dinov2-base 0.8887 at 4× the FLOPs
     - resnet50d-224 0.8842 at 0.35× the FLOPs but the same 185 s/epoch
   - radiodino-s16 is 5× cheaper per study but 0.066 below.
2. **Grand-recipe cost on that trunk:**
   - Row 1 measured 185 s/epoch at batch 12, so 36 epochs take ≈ 111 min (1.85 h).
   - 2 h buys ≈ 38 epochs ≈ 11.3k optimizer steps (291 steps/epoch).
   - Features that slowed row 4 (EMA, fold −1 at +25 % studies) push this toward 36 × ~300–470 s ≈ 3–4.7 h.
   - At the 60 % VRAM cap, batch 12 needs the allocator gap closed. Otherwise use batch 4 × acc 3 as in row 3: 7.5 GB allocated, 205 s/epoch, 0.9045.
3. **The cheap probe worth pricing is dinov2-small 224/12 drop40.**
   - It costs 45 s/epoch and 132 GFLOPs/study (3.7× cheaper), so 36 epochs take 27 min and 2 h buys about 160 epochs.
   - At 24 epochs it read 0.8865, which is −0.018 from the seat, outside both the 0.01 band and the 1.5 % bar.
   - It is a curriculum and horizon test bed, not a seat, until a longer horizon closes that gap. P(closes) is unmeasured.
