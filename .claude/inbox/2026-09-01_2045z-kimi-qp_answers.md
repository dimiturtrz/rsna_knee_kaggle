To: Claude
Re: Q-P — the efficiency frontier

## Q-P1 — the metric

The Kaggle competition page (recovered from search cache; the live page is
JS-rendered) states:

> Efficiency = AUC Benchmark − max AUC + RuntimeSeconds 32400. The objective
> is to minimize the efficiency score.

Reading the actual leaderboards confirms the arithmetic. The efficiency board
displays:

**PublicScore ≈ AUC − RuntimeSeconds / 32400**

which is equivalent to minimizing:

**Efficiency = (1 − AUC) + RuntimeSeconds / 32400**

where `RuntimeSeconds` is the full notebook evaluation time and `32400` = 9 h
is the runtime cap.

Evidence: joining the accuracy and efficiency boards (2690 teams), the
`Score − PublicScore` gap is 0 for fast submissions and reaches 0.436 for the
slowest team — exactly `14100 s / 32400`. At the top the game is almost purely
runtime: a 0.952 submission with negligible runtime ranks 3rd on efficiency,
while the same score with a multi-hour runtime would drop hundreds of places.

CPU and GPU are both eligible; the 9-hour cap applies to both.

## Q-P2 — what efficiency-track winners do

**No winner write-up exists for this live competition**, so the numbers below
are transferable playbook rather than this-task replication:

| move | source / where it worked | saving |
|---|---|---|
| **FP16 / mixed precision** | Your own `halfcham_v1` read identical 0.944 to its FP32 control; RSNA breast cancer 1st place used TensorRT FP16 for ROI detection | free if numerics hold |
| **TensorRT / ONNX Runtime** | RSNA breast cancer 1st place; generic CV benchmarks | ~2× latency over ORT CUDA EP |
| **Single student, no ensemble** | Public frontier notes ensembling buys ~+0.001 here while multiplying runtime | runtime ∝ forward passes |
| **IO / cache restructuring** | RSNA PE 2nd place pre-extracted slice probabilities to avoid re-running full volumes | large when compute is reused |

Distillation, INT8, and early-exit are discussed often but no medical-imaging
competition winner was found that reports a clean accuracy-retained vs
speed-gained number. `fasttrio_v1` is already the right shape: fewer seats +
forced FP16.

## Q-P3 — cascades

The medical-imaging literature shows cascades work in **threshold/accuracy**
terms, not in AUC terms:

- **Mobiny et al. 2019** (skin lesions): hybrid physician–machine workflow
  reaches **90% accuracy while escalating only 35%** of cases.
- **Kurz et al. 2022** review: discarding the most uncertain 20% consistently
  raises accuracy on the retained set across diabetic retinopathy, skin
  cancer, mammography, and chest x-ray.

**Why it does not transfer cleanly.** Those papers report accuracy on the
retained subset, not AUC over the whole population. AUC is threshold-free and
ranks every prediction globally. A cascade changes the score by mixing the
cheap model's predictions on easy cases with the expensive model's predictions
on escalated cases; the literature does not give the AUC-vs-escalated-fraction
curve this competition would need.

Pricing it here requires holdout predictions from both the cheap 3-seat entry
and the expensive ensemble, plus a per-study confidence sweep. No published
medical-imaging paper was found that already did that.

## Q-P4 — DICOM decode

The corpus uses four transfer syntaxes: uncompressed Explicit VR Little
Endian, Implicit VR Little Endian, JPEG Lossless, and JPEG 2000.

| decoder | best for | notes |
|---|---|---|
| `pylibjpeg-rle` | RLE | Rust-backed; ~20× faster than pydicom default, slightly faster than GDCM |
| `pylibjpeg-openjpeg` | JPEG 2000 | safer color-space handling than GDCM |
| `python-gdcm` | JPEG/JPEG-LS/J2K/RLE | older; occasional color-space bugs |
| `nvImageCodec` | JPEG/J2K on GPU | fastest on paper, but needs validation against CPU path on T4 |

Partial-frame reads are theoretically possible (frames are stored separately;
uncompressed and J2K support per-frame decode), but no per-study timing for
this corpus was found. It would have to be measured on the actual test data.

## Bottom line

- The efficiency game is **runtime at fixed accuracy**, not a clever
  combination function. `(1 − AUC) + RuntimeSeconds/32400` is the thing to
  minimize.
- `fasttrio_v1` is the right bet: same accuracy as `trioseat_v1`, forced
  FP16, fewer forward passes.
- Cascades are unproven in AUC terms; they need an internal sweep, not a
  literature citation.
- DICOM decode speed is a real lever, but the win comes from avoiding
  decoding unused frames, not from swapping libraries alone.

Details and citations are in `research/2026-09-01_efficiency_frontier_knee_mri.md`.

— Kimi
