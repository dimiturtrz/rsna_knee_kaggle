# The efficiency frontier — Q-P

Gathered 2026-09-01 for Claude's Q-P about the RSNA Knee Abnormality Detection
**efficiency prize** track. Four questions: the exact metric, what Kaggle
efficiency-track winners do, cascades, and DICOM decode speed.

## Q-P1 — the efficiency metric

The Kaggle competition page gives the formula in a snippet that survives in
search caches (the live page is JS-rendered and the static HTML is empty):

> Efficiency = AUC Benchmark − max AUC + RuntimeSeconds 32400. where AUC is ...
> The objective is to minimize the efficiency score.

Reading the leaderboard data confirms the intended arithmetic. The efficiency
board's `PublicScore` equals the submission's accuracy-board `Score` minus a
runtime penalty:

**Displayed efficiency score ≈ AUC − RuntimeSeconds / 32400**

Equivalently, the quantity Kaggle asks competitors to **minimize** is:

**Efficiency = (1 − AUC) + RuntimeSeconds / 32400**

where `RuntimeSeconds` is the full notebook evaluation time and `32400` is the
9-hour runtime cap.

**Evidence from the board.** Joining the accuracy and efficiency leaderboards
(2690 teams present on both) shows the efficiency score is almost always at or
below the accuracy score, with the gap growing for slow submissions:

| statistic | value |
|---|---|
| median `Score − PublicScore` | 0.000 |
| 75th-percentile gap | 0.001 |
| maximum gap | 0.436 |
| implied runtime at max gap | ~14,100 s (~3.9 h) |

A gap of 0.436 exactly matches a runtime penalty of `0.436 = 14100 / 32400`.
The top efficiency ranks are held by teams whose PublicScore is within a few
parts per thousand of their accuracy score — i.e. they run fast enough that
the penalty is negligible.

**Implication.** At the top of the accuracy board the game is almost purely
runtime. A team with AUC 0.952 and negligible runtime ranks 3rd on efficiency;
a team with the same 0.952 but a multi-hour runtime would drop hundreds of
places. Conversely, a cheap 0.938 run can outrank slower 0.944 runs, which is
exactly the `fasttrio_v1` bet.

## Q-P2 — what Kaggle efficiency winners actually do

**There is no winner write-up for this live competition.** The efficiency track
is new to this RSNA challenge, so no past solution exists for this exact task.
What follows is the transferable playbook from Kaggle efficiency prizes in
other domains and from past RSNA solutions where runtime was a constraint.

### Strategies with evidence

| strategy | source / where it worked | measured saving |
|---|---|---|
| **FP16 / mixed-precision inference** | RSNA breast cancer 1st place used TensorRT FP16 for the YOLOX-nano ROI detector; this repo's own `halfcham_v1` read the same 0.944 as its FP32 control | free if numerics hold |
| **TensorRT / ONNX Runtime** | RSNA breast cancer 1st place; general CV benchmarks report ~2× over ORT CUDA EP with FP16 | ~2× latency reduction |
| **Single small student instead of ensemble** | The public frontier notes ensembling buys only ~+0.001 on this board while multiplying runtime | runtime ∝ number of forward passes |
| **Decode/IO restructuring** | RSNA PE 2nd place pre-extracted heart-slice probabilities and feature caches to avoid repeating full-volume inference | large when compute is reused |

### Strategies often claimed, less often quantified

- **Knowledge distillation** to a smaller model: common in generic efficiency
  literature, but no medical-imaging competition winner was found that reports
  the exact accuracy retained vs speed gained.
- **Early-exit / cascades**: discussed below; the medical-imaging numbers are
  mostly from retrospective papers, not competition solutions.
- **Quantization to INT8**: TensorRT supports it, but the risk of AUC loss on
  small positive classes (Fracture has 68 cases) is high and would need a
  witness check before it could be shipped.

**Bottom line for this track.** The biggest lever is not a clever metric
manipulation — it is simply **running fewer models at lower precision with
faster IO**. The `fasttrio_v1` design (three same-pixel-group seats in FP16
under `LogitAverage`) is already the right shape.

## Q-P3 — cascades: the medical-imaging evidence

A cascade routes easy cases to a cheap model and escalates only uncertain
cases. The medical-imaging literature shows the concept works, but the
reported numbers are almost all in **accuracy / threshold terms**, not in
AUC — which is what this competition ranks on.

### Published numbers

- **Mobiny et al. 2019** (*Journal of Clinical Medicine*, skin lesion
  diagnosis): a hybrid physician–machine workflow reaches **90% accuracy while
  referring only 35% of cases to physicians**. The model itself (MCDO over
  VGG/ResNet/DenseNet) is the cheap tier; the physician is the expensive tier.
  This is a cascade in effect, even if not named as one.

- **Kurz et al. 2022** (*JMIR*, systematic review of uncertainty estimation in
  medical image classification): across diabetic retinopathy, skin cancer,
  mammography and chest-x-ray studies, discarding the most uncertain 20% of
  predictions consistently improves accuracy on the retained set. Leibig et
  al. 2017 (diabetic retinopathy) is one of the primary sources: MCDO-based
  uncertainty filtering raises accuracy when ≥20% of cases are deferred.

- **Ghesu et al. 2019/2021** (chest radiograph / multi-modality): discarding
  uncertain cases improves AUROC on the retained fraction, but the reported
  gain is on a *retained subset*, not on the whole population including the
  escalated ones.

### Why it does not transfer cleanly to this competition

The literature reports **accuracy on retained cases**, not AUC over all cases.
AUC is threshold-free and ranks predictions globally. A cascade that escalates
30% of studies changes the score in two ways:

1. The cheap model's predictions on the escalated 30% are replaced by the
   expensive model's predictions.
2. The expensive model's errors are mixed into the final score.

The literature does not give a closed-form or empirical curve for how AUC
changes as a function of the escalated fraction. The closest conceptual result
is that the **cheap model's confidence must rank errors well** — otherwise the
escalated set contains too many already-correct cases and the gain is lost.

For this competition, the practical pricing would require:
- holdout predictions from both the cheap 3-seat entry and the expensive
  ensemble;
- a per-study confidence (e.g. mean entropy, max predicted probability);
- a sweep of escalation thresholds and the resulting blended AUC.

No published medical-imaging paper was found that reports this exact curve.

## Q-P4 — the fastest safe DICOM read path

### What the data look like

The official data description (confirmed in community documentation) says the
569.76 GB corpus uses **four transfer syntaxes**:

- uncompressed Explicit VR Little Endian
- JPEG Lossless
- JPEG 2000
- Implicit VR Little Endian
- RLE is also possible via the standard DICOM toolkit ecosystem

A single competition kernel must handle all of them.

### Decoder options

| decoder | coverage | notes |
|---|---|---|
| `pydicom` default | uncompressed, RLE lossless, deflated | no JPEG/J2K by default |
| `pylibjpeg-libjpeg` | JPEG 8-bit lossy/lossless | pure-ish Python wrapper around libjpeg |
| `pylibjpeg-openjpeg` | JPEG 2000 | wrapper around OpenJPEG |
| `pylibjpeg-rle` | RLE | Rust-backed, fast |
| `python-gdcm` | JPEG, JPEG-LS, JPEG 2000, RLE | older, sometimes color-space bugs |
| `nvImageCodec` | JPEG, JPEG 2000, HTJ2K | GPU-accelerated, Kaggle T4 compatible |

### Benchmarks

`pylibjpeg-rle` ships its own benchmark (per 1000 encodes/decodes):

| dataset | pydicom default | pylibjpeg-rle | python-gdcm |
|---|---|---|---|
| OBXXXX1A.dcm (480 kpix) | 30.6 s | 1.4 s | 1.5 s |
| MR_small.dcm (4 kpix) | 3.0 s | 0.11 s | 0.29 s |
| SC_rgb.dcm (10 kpix) | 1.9 s | 0.11 s | 0.21 s |

For RLE, `pylibjpeg-rle` is ~20× faster than pydicom default and slightly
faster than GDCM.

For JPEG 2000 — the likely slowest path — there is no published head-to-head
benchmark in this exact corpus. The safe choices are `pylibjpeg-openjpeg` or
`python-gdcm`; `nvImageCodec` is the fastest on paper but requires a CUDA
setup and validation that the decoded array matches the CPU path.

### Partial reads

The repo already samples only the band (0.1–0.9) and a fixed number of slices.
The next optimization is to **not decode frames that will be discarded**:
`pydicom` reads the whole pixel array by default, but the DICOM file format
stores frames contiguously and a low-level reader can seek to and decode only
the InstanceNumbers in the chosen band. Whether this wins depends on transfer
syntax: for uncompressed data the seek is trivial; for JPEG 2000 each frame is
independently decodable, so partial decode is possible; for RLE the bitstream
is frame-based and also supports it.

No published per-study timing for this corpus was found; any number would need
to be measured on the actual test set.

## Summary for Claude's four questions

1. **Metric.** Minimize `(1 − AUC) + RuntimeSeconds / 32400`. The efficiency
   board displays `AUC − RuntimeSeconds / 32400`. Runtime is the full notebook
   evaluation; 32400 s = 9 h cap.

2. **Winner runtime strategies.** No winner exists yet. The transferable
   moves are FP16 inference, TensorRT/ONNX, single-student instead of
   ensemble, and IO restructuring. Quantified savings are mostly generic
   (~2× from TensorRT FP16) rather than from a medical-imaging winner.

3. **Cascades.** Medical imaging papers show cascades/uncertainty filtering
   improve retained-set accuracy (Mobiny: 90% accuracy at 35% escalation;
   Kurz review: gains from discarding the most uncertain 20%). No published
   result gives the AUC-vs-escalated-fraction curve needed for this
   competition.

4. **DICOM decode.** Use `pylibjpeg-*` plugins (Rust-backed RLE is ~20×
   faster than pydicom default); for JPEG 2000 prefer `pylibjpeg-openjpeg` or
   `python-gdcm`, with `nvImageCodec` as a GPU option if validated. Partial
   frame reads are theoretically possible but unmeasured on this corpus.

## Sources

- Kaggle competition page, efficiency metric snippet (JS-rendered; formula
  recovered from search cache + leaderboard join).
- Downloaded leaderboards 2026-09-01: accuracy public board and
  `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` `full_leaderboard.csv`.
- Mobiny A, Singh A, Van Nguyen H. "Risk-aware machine learning classifier
  for skin lesion diagnosis." *J Clin Med* 8(8):1241 (2019).
- Kurz A, et al. "Uncertainty estimation in medical image classification."
  *JMIR* (2022). [PMC9382553](https://pmc.ncbi.nlm.nih.gov/articles/PMC9382553/)
- Leibig C, et al. "Leveraging uncertainty information from deep neural
  networks for disease detection." *Sci Rep* 7:17816 (2017).
- `pylibjpeg-rle` benchmarks:
  [GitHub](https://github.com/pydicom/pylibjpeg-rle).
- pydicom plugin table:
  [pydicom docs](https://pydicom.github.io/pydicom/dev/guides/plugin_table.html).
- NVIDIA nvImageCodec DICOM sample:
  [docs](https://docs.nvidia.com/cuda/nvimagecodec/samples/DICOM.html).
