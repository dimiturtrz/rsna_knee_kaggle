# The scanner axis exists in the data and not in the ruler

Measured 2026-09-05, card-free, from the DICOMs and from `canonical_metadata.csv`.

## The census — our corpus is genuinely multi-vendor

Over all 4,407 studies, 0 unreadable:

| vendor | studies | spelling on disk |
|---|---|---|
| Siemens | ~1,951 | Siemens Healthineers 1,053 · SIEMENS 804 · Siemens 94 |
| Philips | ~1,301 | Philips Medical Systems 718 · Philips 492 · Philips Healthcare 91 |
| GE | ~905 | GE MEDICAL SYSTEMS 868 · GEHC 37 |
| Toshiba / Canon | 226 | |

Field strength: **1.5T 2,543 · 3T 1,601 · absent 238**. Forty-plus distinct `ManufacturerModelName`
values, the largest being Ingenia 741, MAGNETOM Vida 447, Aera 400, Achieva dStream 327, MAGNETOM
Avanto fit 262, SonataVision 214, SIGNA Explorer 213, SIGNA Artist 208, Optima MR450w 158, SIGNA
EXCITE 136, Vantage 129, MAGNETOM Aera 122.

Three vendors above 900 studies and a 61/39 field-strength split is not a nuisance tail. It is a
multi-vendor corpus that we have been treating as one domain.

The vendor spellings do NOT normalise themselves — `SIEMENS`, `Siemens` and `Siemens Healthineers`
are three strings for one manufacturer, and `GE MEDICAL SYSTEMS` collides on prefix with `Philips
Medical Systems`. Any grouping has to canonicalise on a vendor map, not on the raw tag.

## The defect — our reader drops both fields

`StudyMetadata` (`core/data/corpus/study_metadata.py`) declares `field_strength_t: float | None` and
`scanner: str | None`. The fastMRI, MRNet and Rijeka builders populate them. **Our own corpus does
not**: `canonical_metadata.csv`, rebuilt today via `build_cache --metadata-only`, holds **0 non-null of
4,407** in both columns, while a direct `pydicom` read of a single study returns
`Manufacturer = Philips Healthcare`, `ManufacturerModelName = Ingenia`, `MagneticFieldStrength = 1.5`.
The tags are present in the files and `Study.from_directory(...).metadata()` does not carry them out.

Same table, adjacent gaps worth recording: `pixel_spacing_mm` 344/4,407 non-null and `slice_spacing_mm`
855/4,407. `InstitutionName`, `StationName` and `DeviceSerialNumber` are genuinely ABSENT from the
files — those are not reader gaps and no site-level grouping is available.

Filed as knee-m4jkz.

## Why this matters — the ruler has no vendor axis

The 909-study holdout is a **random** split over the 4,407. Train and holdout therefore share the
scanner distribution by construction. Every number this project has ever quoted — every member score,
every greedy seat order, every ensemble delta — was measured on a ruler that cannot see acquisition
shift, because it contains none.

The Kaggle test set need not share that mix, and nothing we hold tells us whether it does.

A ledger grep for `grouped split|site split|scanner|manufacturer|ood holdout|distribution shift`
returns **zero hits** across `experiments/README.md`. This axis has never been touched here.

## The prior art says do not reach for normalization

`research/2026-09-05_cross-vendor-domain-shift-prior-art-cardiac-seg.md` records five independent
confirmations from the sibling cardiac-seg project that harmonization/normalization does not move a
cross-vendor result — Nyul null at -0.007, oracle blood-anchor null, the whole norm grid flat inside
0.019 of single-seed scatter, N4 and bias-field augmentation both regressing. We already z-score and
already augment intensity, and their one-direction rule forbids stripping an axis we diversify.

So the finding here is NOT "build a scanner normalizer". It is that **we own an unmeasured axis and a
free instrument for it**.

## The instrument, and why it costs no card

`Fitting` persists the kept epoch's per-study holdout probabilities beside every checkpoint as a
`HoldoutRecord` (`kneemri/runs/holdout_record.py:115`), keyed by study NAME rather than position.
`record_mix` reaches 104 such records over the shared 909 studies. Per-vendor and per-field-strength
AUC for every seated member is therefore **arithmetic over files already on disk** — no GPU, no corpus
rebuild, no competition with whatever is on the card.

The question it answers is not "which vendor scores worst" — a vendor subset is smaller than 909 and
its noise floor is correspondingly wider than the 0.0062 we quote, so absolute per-vendor AUCs will not
be individually significant. The question is **whether member RANK reorders across vendors**. Greedy
seat selection consumed one ordering; if that ordering is mix-dependent, then ensemble selection has
been fitted to our scanner mix, and that is a named, testable mechanism for the seven board flats —
holdout gains that stop transferring. Rank agreement across three vendor subsets of the SAME 909
studies is a paired comparison and is far better conditioned than the per-vendor absolutes.

cardiac-seg supplies the matching mechanism from the other side: their 4-seed deep ensemble bought
+0.000 on the held-out vendor, with BALD putting only 15-18% of that uncertainty as epistemic. A
same-recipe ensemble is structurally blind to the shift it was not selected against.

Filed as knee-v9m5z. Two consequences already filed beside it: knee-vm3je (re-run greedy seat
selection against a worst-vendor or across-vendor objective, also card-free) and knee-hotxj (condition
the model on vendor and field strength, gated on the ruler showing the axis carries signal at all).
The crop invariant is a possible confound in that ruler and is filed as knee-mugpz: `SlotReader.window`
returns None with no spacing, so `crop_mm=130` holds mm/pixel constant only on the subset where the
window applies, and that subset has never been measured.

## What is closed, so this is not read as a fresh start

Everything else has been walked. Architecture and capacity: DINOv2-small 0.8512 against base 0.8455
(inside floor, batch handicap already removed) and large 0.8048; ~25 foreign trunks all below the
natural-image control, `swin_supervised` 0.8331 and `cnx_dense` 0.7941 among them, which is also the
public frontier's answer. Depth: `band_deep_matched` reading now. SSL/MAE: five arms, 0.6735-0.7309
against 0.8578. Site count: `merged_corpus` +0.0043 for a second site, `merged3_corpus` -0.0024 for a
third, and a ROTATION rather than an addition. Data volume: concave, the next doubling priced ~+0.009.
Epoch budget: `convergence_check` over 214 runs finds 166 converged and all 8 undertrained runs are
4-epoch pretraining probes, not seated members.

That leaves generalization, which is precisely the thing an in-distribution ruler cannot see.
