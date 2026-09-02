# Report-to-label audit against the 58 radiologist-annotated studies

Assignment: `knee-skcf`.  Compare the current report-derived labels
(`report_labels_v2.csv`) with the 58 studies in `train.csv` that carry
all twelve image-read labels.

## Method

- Gold positives: image-read label `== 1`.
- Extracted positives: report value `> 0.5` (the three YES grades
  `0.68 / 0.82 / 0.94`; verdict `YES` gives the same set).
- A value of `0.28` (verdict `UNK`) is treated as non-positive.
- Missing extracted rows are treated as extraction-absent, i.e. non-positive.
- Source paths:
  - `D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv`
  - `D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/report_labels_v2.csv`

## Coverage

- 58 gold-annotated studies in `train.csv`.
- 57 of them have a row in `report_labels_v2.csv`.
- 1 study is missing from `report_labels_v2.csv`:
  `1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220`.
  It is gold-positive for **ACL, PF OA, Effusion, Contusion, Fracture**.
  Those five positives are therefore counted as missed in the table below.

## Per-label confusion table (n = 58)

| label | gold pos | gold neg | extr pos | agree | missed | denied | FN rate | FP rate | agree rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 24 | 34 | 32 | 48 | 1 | 9 | 4.2% | 26.5% | 82.8% |
| MCL | 9 | 49 | 18 | 49 | 0 | 9 | 0.0% | 18.4% | 84.5% |
| Medial Meniscus | 26 | 32 | 32 | 50 | 1 | 7 | 3.8% | 21.9% | 86.2% |
| Lateral Meniscus | 23 | 35 | 30 | 45 | 3 | 10 | 13.0% | 28.6% | 77.6% |
| Medial OA | 15 | 43 | 22 | 49 | 1 | 8 | 6.7% | 18.6% | 84.5% |
| Lateral OA | 11 | 47 | 16 | 49 | 2 | 7 | 18.2% | 14.9% | 84.5% |
| PF OA | 21 | 37 | 24 | 45 | 5 | 8 | 23.8% | 21.6% | 77.6% |
| Effusion | 35 | 23 | 50 | 41 | 1 | 16 | 2.9% | 69.6% | 70.7% |
| Synovitis | 27 | 31 | 16 | 41 | 14 | 3 | 51.9% | 9.7% | 70.7% |
| Baker's | 12 | 46 | 16 | 52 | 1 | 5 | 8.3% | 10.9% | 89.7% |
| Contusion | 19 | 39 | 36 | 35 | 3 | 20 | 15.8% | 51.3% | 60.3% |
| Fracture | 18 | 40 | 19 | 47 | 5 | 6 | 27.8% | 15.0% | 81.0% |

## Systematic error directions

**Over-called (extractor positives that the radiologist denied)**

- **Effusion** is the worst: 16/23 gold-negatives are called positive.
  A report phrase such as "small effusion" or "trace fluid" appears to be
  read as a positive by the extractor, while the annotator requires a
  moderate or large distension.
- **Contusion** is next: 20/39 gold-negatives are called positive.
  Bone-marrow edema language and fracture-adjacent signal likely trigger
  this.
- **Menisci, ACL/MCL, OA compartments** show a steady ~15–30% false-positive
  rate; the extractor is more willing than the annotator to call a finding
  present from report text.

**Under-called (extractor misses a radiologist-positive finding)**

- **Synovitis** is the only large under-call: 14/27 gold-positives are
  missed.  The report may describe synovitis implicitly (effusion,
  inflammatory change) or not use the word at all.
- **Fracture** and **PF OA** miss ~1 in 4 positives; the missing extracted
  row contributes one miss to each of ACL, PF OA, Effusion, Contusion,
  Fracture.

## Comparison with the per-finding calibration offsets

The fitted offsets in `core/data/labels/targets.py` (negative = over-call,
positive = under-call) line up for the extremes:

- Contusion offset `−0.531` ⇢ heavy over-call (audit FP 51%).
- Synovitis offset `+0.985` ⇢ heavy under-call (audit FN 52%).

The one clear tension is **Effusion** (offset `+0.432`, implying under-call),
while the audit shows it as the strongest over-call (FP 70%).  Either the
offset was fitted on a different slice of the annotation table, or the
extractor's effusion threshold has drifted since the offsets were computed.

## Go / no-go on parser improvement

**Go**, but with a narrow scope.

The current labels are soft targets with a learned calibration, so some of
this noise is already absorbed at training time.  A parser rewrite trained
on the same 58 studies risks overfitting the same annotations and may not
move the held-out AUC.  However, three things are worth fixing:

1. **The missing study** — one row in `report_labels_v2.csv` is absent for
   a fully-annotated study.  This is a pipeline defect, not a parser error,
   and it silently removes five gold positives from training.
2. **Effusion threshold / size language** — the largest single error mode.
   A rule or small classifier that distinguishes trace/small from
   moderate/large effusion would recover many false positives.
3. **Synovitis recall** — the largest under-call.  Mapping implicit
   inflammatory descriptions to synovitis would help, though the criteria
   explicitly require synovial thickening, so the gain is capped.

The other findings are within the noise expected from report-versus-image
annotation disagreement.  A broad parser rewrite is not justified by this
audit; targeted fixes on the three items above are.
