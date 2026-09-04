# 58-vs-57 study denominator mismatch

## Finding

The 58 radiologist-adjudicated gold studies used by `local_reader_annotated.csv` and `report_labels_qwen38_annotated.csv` are the same 58 studies. However, the project's default label table `report_labels_v2.csv` contains only **57** of them.

## Which study is missing

`1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220`

- Present in `train.csv` (row 3412 of 4,407).
- Present in `report_labels_v4.csv` (4,407 studies).
- **Absent** from `report_labels_v2.csv` and `report_labels_v3.csv` (both 4,406 studies).
- The report text is present and appears normal (~1,858 characters, English).

## Implication

Any score quoted "on the 58 gold studies" against `report_labels_v2.csv` is actually on **57 studies**, unless the intersection is explicitly taken. The missing study is included in v4, so v4 is the table that covers the full 58-study gold set.

## Files checked

| file | studies | missing study present? |
| :-- | ---: | :--- |
| `raw/rsna_knee_abnormality/train.csv` | 4,407 | yes |
| `processed/rsna_knee_abnormality/labels/report_labels_v2.csv` | 4,406 | **no** |
| `processed/rsna_knee_abnormality/labels/report_labels_v3.csv` | 4,406 | **no** |
| `processed/rsna_knee_abnormality/labels/report_labels_v4.csv` | 4,407 | yes |
| `processed/rsna_knee_abnormality/labels/local_reader_annotated.csv` | 58 | yes |
| `.agent-runtime/scratch/report_labels_qwen38_annotated.csv` | 58 | yes |

## Recommendation

Before quoting any delta "on 58 studies", verify whether the denominator is 58 or 57. If the ruler is `report_labels_v2.csv`, the honest denominator is 57 unless the missing study is added or v4 is used.
