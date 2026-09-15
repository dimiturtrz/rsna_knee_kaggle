# ROI locator audit — sagittal, SKM-TEA -> RSNA (knee-vgz5b.3 / .4)

Offline-only: the SKM-TEA licence ruling (knee-vgz5b.7) is pending, so nothing here ships.

## What was built

- `core.data.corpus.skmtea_slices` — SKM-TEA DESS slices streamed from `dicoms/MTR_*.tar.gz` in memory
  (no extraction; the dataset has no h5 images), both echoes, every third slice of the 0.1-0.9 band, 336 px.
  Split is SKM-TEA's own v1.0.0 annotation split: 86 / 33 / 36 scans = 3698 / 1419 / 1543 slices.
- `kneemri.models.locator` — DINOv2-small, patch tokens from layers 4, 8, 12, light upsampling conv head,
  5 classes (medial/lateral tibial cartilage and menisci merged: one sagittal slice cannot tell its side).
- `kneemri.training.locator_fit` — 40 epochs, weighted CE + soft Dice, crop/flip geometry and a random
  nine-knot intensity lookup table.
- `kneemri.evaluation.locator_audit` — landmark distances on SKM-TEA test, and RSNA overlay grids.

## SKM-TEA test split (36 scans, 1543 slices), committed-code run

| tissue | Dice | centroid median mm | p90 mm | slices missed / present | extra slices |
|---|---|---|---|---|---|
| patellar cartilage | 0.398 | 3.0 | 6.6 | 109 / 661 | 74 |
| femoral cartilage | 0.346 | 5.8 | 18.1 | 90 / 1164 | 54 |
| tibial cartilage | 0.396 | 3.2 | 6.0 | 146 / 938 | 123 |
| meniscus | 0.362 | 3.6 | 8.8 | 96 / 934 | 183 |

The prototype of the same recipe read 0.402 / 0.344 / 0.398 / 0.361, so the port reproduces it.
"Present" means more than 20 true pixels on the slice; distance is per slice between the true and the
drawn region centroids at 0.3125 x 512 / 336 mm per pixel.

Dice misses the bead's 0.6 meniscus bar. Read why before reading that as failure: cartilage is a few
pixels thick at 336 px and the head sees a 24 x 24 token grid, so boundary error dominates Dice, while
the question `.5` asks is WHERE the tissue is. The centroid lands within ~3-6 mm median, which is the
number an ROI crop consumes. Femoral p90 18 mm is the long curved plate whose centroid moves with what
part of it is drawn, not a location miss.

## RSNA sagittal overlays (50 studies per slot, rng seed 1; tiles at slices 5, 10, 15 of 20)

Read by eye, prototype checkpoint of the same recipe:

| slot | tiles blank | studies with no joint-line placement | clear misplacements |
|---|---|---|---|
| SAG_T1 | 6 / 150 | 0 / 50 | 0 / 50 |
| SAG_FLUID_FS | 0 / 150 | 0 / 50 | 0 / 50 |
| SAG_FLUID_NOFS | 0 / 30 (10 studies read) | 0 / 10 | 0 / 10 |

Spot-read of the committed-code checkpoint (SAG_T1 grids 0 and 30): 20 / 20 studies placed at the joint
line, 2 / 60 tiles blank, both on off-centre or blurred slices of a study located on its other tiles.

Without the lookup-table augmentation, about 13 of 30 SAG_T1 tiles came back blank: the DESS-only
training contrast did not transfer to T1 until contrast order stopped being a usable cue.

Miss rate at study level is 0 / 110 read against the bead's <=10% bar, so `.4` passes for sagittal.

## Not done

- Coronal and axial planes (stage B: fastMRI+ boxes; stage C: cross-plane via DICOM geometry).
- Medial/lateral separation from slice position in the stack.
- A per-study ROI box table, which `.5` needs to consume.

## 2026-09-15 addendum — the first ROI arm read two confounds, not the idea (knee-vgz5b.10)

`roi_labelwise_drop40_long_s2026` read 0.8984 at ledger75 against its no-ROI control `drop` 0.9051
(909 rows). The per-label deltas pointed at the small structures the crop existed for: Fracture -0.023,
MCL -0.024, ACL -0.012, PF OA -0.010, Lateral OA +0.013. Before calling the thread dead, two harness defects:

- **Resolution (DATA).** The `roi` crops were upsampled from the 336 cache at 0.387 mm/px. RSNA sagittal
  PixelSpacing is p10/p50/p90 0.212/0.312/0.494 mm over 200 series, 73% finer than the cache, so an 87 mm
  median box discarded ~1.24x linear detail and interpolated blur replaced it. Header spacing may itself be
  interpolated; the figure is an upper bound on real detail.
- **Centring (TEST, for the older multiscale verdict).** The Otsu tissue centre `crop80_tissue` cropped
  about sits a median 15.6 mm (p90 29.3) from the locator box centre over 400 sagittal slots; overlays show
  80 mm tissue-centred crops clipping the patella and anterior joint. The multiscale retirement of
  2026-08-16 was read on that centre.

Also measured: locator Dice 0.365 test / 0.483 train on SKM-TEA — boundaries underfit, placement fine
(0 / 110 study-level misses above). The box, not the mask, is what the ROI views consume.

`roi_native` fixes the resolution axis alone: the same boxes cut from the stack decoded at 672 px (0.193
mm/px) and resized down. Built 4407 studies, 0 decode failures; canonical slots byte-identical to `roi`;
sagittal crops correlate p5 0.9987 / p50 0.9995 with the upsampled ones over 40 studies — so the
difference is fine detail only, and a flat reading will say detail was not what the arm lacked.
