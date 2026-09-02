# RSNA label tables — pseudo-label audit

## What is on disk

`D:/data/volumetric/mri/raw/knee_label_tables/` holds community-derived labels for the 4,407 RSNA training studies. There is no extra imagery; these are **alternative targets** for the same competition training set.

| source | files | rows | type |
|---|---|---|---|
| `blend_*.csv` (6 files) | `blend_tb10.csv`, `blend_tb25.csv`, `blend_tb5.csv`, `blend_three.csv`, `blend_tiebreak.csv`, `blend_two.csv` | 4,406 | hard-ish community blends: probability + confidence + verdict (YES/NO/UNK) per label |
| `steven_v4_quantised5.csv` | one file | 4,407 | quantised soft targets (12 probability columns) |
| `gxpixel_knee-consensus-labels/` | `train_labels_final.csv`, `labels_prevsharp.csv`, `labels_labelmodel.csv`, `labels_sharp.csv`, `labels_control_steven.csv`, `folds_v1.csv`, `gold_58.csv` | 4,407 | consensus of 4-5 extractors, with frozen site-grouped folds and a 58-study gold holdout |
| `rayanbabur_rsna-knee-2026-calibrated-soft-targets/` | `train_targets_soft.csv`, `train_targets_hard.csv`, `train_targets_weights.csv`, `gold_benchmark_58.csv` | 4,407 | calibrated soft targets + confidence weights, validated against the same 58 gold studies |
| `flight0234_rsna-knee-hybrid-report-labels/` | `report_labels_v4hybrid.csv` | 4,407 | hybrid report labels |
| `lixin73_rsna-knee-llm-report-labels-sol56/` | two CSVs | 4,407 | GPT-4o/GLM-derived LLM labels |
| `stevenleehans_rsna-knee-llm-report-labels/` | three CSVs | 4,407 | another LLM label family |
| `yunusgmsoy_rsna-knee-llm-labels-4-source-merged/` | `report_labels_v5.csv` | 4,407 | 4-source merged labels |

## Verdict distributions (blend files)

All six `blend_*.csv` files share **identical YES/NO/UNK verdicts**; they differ only in the probability values (mean absolute difference ~0.059 between `tb10` and `tb25`). One study present in the other tables is missing from the blends (`1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220`).

Per-label verdict counts on 4,406 studies:

| label | YES | NO | UNK | mean confidence |
|---|---|---|---|---|
| ACL | 1,226 | 2,825 | 355 | 0.813 |
| MCL | 767 | 3,209 | 430 | 0.789 |
| Medial Meniscus | 2,347 | 1,809 | 250 | 0.858 |
| Lateral Meniscus | 1,121 | 2,843 | 442 | 0.795 |
| Medial OA | 1,619 | 1,662 | 1,125 | 0.682 |
| Lateral OA | 1,087 | 1,862 | 1,457 | 0.610 |
| PF OA | 1,991 | 1,600 | 815 | 0.747 |
| Effusion | 2,666 | 1,306 | 434 | 0.832 |
| Synovitis | 580 | 114 | 3,712 | 0.189 |
| Baker's | 1,114 | 1,269 | 2,023 | 0.508 |
| Contusion | 1,297 | 2,193 | 916 | 0.713 |
| Fracture | 314 | 1,606 | 2,486 | 0.406 |

Key observation: **Synovitis and Fracture are mostly UNK** in the blends, and Fracture mean confidence is very low (0.406). These are exactly two of the rare labels where the gold set is enriched with positives, so the blend sources are uncertain.

## Quality of the best-labelled sources

The `gxpixel` consensus and the `rayanbabur` calibrated targets are the only two sources that report validation against the 58 radiologist-labelled gold studies.

- `gxpixel` README claims ~0.90 macro AUC on gold-58 for the consensus, with `labels_prevsharp.csv` and `train_labels_final.csv` tied on ranking and `labels_labelmodel.csv` best on probability calibration.
- `rayanbabur` reports **0.899 macro AUC on soft targets** vs **0.820 on hard-binarised targets** (Δ +0.079). Per-label AUCs range from 0.790 (Synovitis) to 0.987 (ACL).

Both sources keep the 58 gold studies as a held-out evaluation set (`is_gold=1`). The project's own experiments already use `--calibrated-targets` successfully (`calibsix_v1`, 0.941), so the practical value of these tables is partially proven.

## Cross-source agreement

Mean Pearson correlation across the 12 labels on the 4,406 common studies:

| pair | mean r |
|---|---|
| blend vs `rayanbabur` soft | 0.945 |
| blend vs `steven_v4` | 0.919 |
| `gxpixel` train vs `rayanbabur` soft | 0.917 |
| blend vs `gxpixel` consensus | 0.901 |
| `steven_v4` vs `gxpixel` consensus | 0.876 |

The weakest inter-source agreement is on Synovitis, Fracture, and Lateral OA — the rare/noisy labels. For the common labels (ACL, menisci, effusion) agreement is high.

## Competition-rules reading

Kaggle rule 2.6a (copied from `comp/COMPETITION_RULES.txt`):

> You may use data other than the Competition Data ("External Data") to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants ...

The label tables are **publicly available on Kaggle** (community datasets). They are "external data" in the sense that they are not the official competition labels, but they are **not new imagery** — they are re-expressions of the official reports/training set.

Two risks:

1. **Accessibility.** The datasets are public Kaggle datasets. They appear to meet the "publicly available and equally accessible at no cost" test, provided the specific dataset page does not impose extra registration or licence restrictions beyond Kaggle's default.
2. **Derivatives of competition data.** The rules do not explicitly forbid using alternative labels for the same images. The 2nd-place RSNA 2024 solution write-up (Brendan Artley) describes pseudo-labeling from an out-of-fold ensemble as a standard technique. Using a public community label set for the same competition data is lower-risk than privately re-annotating the test set.

The conservative reading: **using these tables as soft training targets is allowed**, but they should be declared as external data in the submission description if Kaggle requires disclosure.

## Recommendations

1. **Best single source for soft training targets:** `rayanbabur_rsna-knee-2026-calibrated-soft-targets/train_targets_soft.csv`. It has the cleanest format (12 soft columns + `is_gold`), reports the strongest gold-58 AUC (0.899), and includes `train_targets_weights.csv` for confidence-weighted loss.
2. **Best source with a frozen CV split:** `gxpixel_knee-consensus-labels/train_labels_final.csv` + `folds_v1.csv`. Use this if you want to run site-grouped cross-validation that matches the consensus author's setup. The `folds_v1.csv` has `fold=-1` for the 58 gold studies.
3. **Best for label-model uncertainty:** `gxpixel_knee-consensus-labels/labels_labelmodel.csv` (Snorkel/Dawid-Skene posterior probabilities, lowest Brier on gold-58).
4. **Avoid for primary training:** the six `blend_*.csv` files. They are missing one study, have low-confidence UNK-heavy verdicts on Synovitis/Fracture, and offer no gold-58 quality report. They might still be useful as an ensemble of label families.
5. **Use `gold_58.csv` only for evaluation.** Both `gxpixel` and `rayanbabur` explicitly flag the 58 gold studies as evaluation-only.

## Next concrete actions

- `knee-b4g6` — verify these label tables load through the existing `--calibrated-targets` pipeline and measure a member trained on `rayanbabur` soft targets vs the current calibrated target source.
- `knee-lkni` — confirm the Kaggle dataset pages for `rayanbabur` and `gxpixel` do not carry NC/ND or registration-restricted licences.
- `knee-icns` — once label-table value is settled, re-run the OrthoFoundation-L trunk-replacement smoke test with the best target source.

## References

- `D:/data/volumetric/mri/raw/knee_label_tables/gxpixel_knee-consensus-labels/README.md`
- `D:/data/volumetric/mri/raw/knee_label_tables/rayanbabur_rsna-knee-2026-calibrated-soft-targets/target_quality_report.md`
- `D:/data/volumetric/mri/raw/knee_label_tables/rayanbabur_rsna-knee-2026-calibrated-soft-targets/target_quality_audit.json`
- `comp/COMPETITION_RULES.txt`, Section 2.6 (External Data and Tools)
- `comp/board.jsonl`, entries `calibsix_v1` and `corpusfull_v1` for the external-data context
- `research/2026-09-01_past_ensemble_plateau_mechanisms.md`, Section 4, for the RSNA 2024 pseudo-labeling precedent
