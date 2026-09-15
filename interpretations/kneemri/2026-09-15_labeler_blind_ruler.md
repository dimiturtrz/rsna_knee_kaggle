# The learned labeler on a ruler it never saw: worse pooled, better on the tear columns

2026-09-15, knee-wko44. CPU only. It answers the circularity in 6bcd6a6a: Qwen v3 read ~0.894 against v2's 0.866 on
gold-58, the ruler the recut table was fitted on.

## The ruler

Blind report-text adjudications under `processed/rsna_knee_abnormality/audit400/`, keys sealed until the answers
were on disk. All were built 2026-09-10/11, after the Qwen sweep (Sep 6), so neither table was tuned on them.

Only the RANDOM-within-stratum packs are used:

| pack | cells | stratum |
|---|---|---|
| root402 | 402 | U 338 / D 64 |
| denom60 | 60 | U |
| classb120 | 120 | U, class-B columns |
| classbd120 | 120 | D |
| dstratum120 | 120 | D, direction and OA balanced |

- **Excluded as selection-biased:** holdout469 (v2 lone dissent, biased against v2), balance56 and imagesel56.
- **Strata:** U means all five tables agree after binarising at >= 0.5 (v2, flight, lixin, steven_blend,
  steven_full); D means they don't. The corpus cell shares are 0.6934 and 0.3066 (`audit400/manifest.json`).
- **Gold-58:** its 13 cells are dropped, leaving **n = 809 cells** (393 in root402).
- **Cut:** every table is binarised at 0.5.

A random sample inside each stratum, reweighted by the stratum's true share, is an unbiased estimate of a table's
accuracy. So the fact that v2 helps define U does not bias the estimate for v2. What it does bias is any pooled
number over the column-oversampled packs, so those are quoted only as paired differences.

## Tables

- **v2:** `labels/report_labels_v2.csv`.
- **qwen_v3:** `.agent-runtime/scratch/report_labels_v3.csv`, md5 e10afbc9. This is the Qwen3.8 criteria-era sweep.
- **recut_v3:** `labels/report_labels_v3.csv`, md5 44e73aa4. This is v2 re-cut by level calibration on gold-58.
  Despite the name, it is not the Qwen reader (knee-clzzs note A).

## Numbers

root402, corpus-weighted, 95% stratified bootstrap (4000 draws):

| table | U acc (n 330) | D acc (n 63) | weighted acc | minus v2, paired |
|---|---|---|---|---|
| v2 | 0.985 | 0.841 | **0.941 [0.911, 0.967]** | — |
| qwen_v3 | 0.885 | 0.889 | 0.886 [0.851, 0.919] | **−0.055 [−0.093, −0.016]** |
| recut_v3 | 0.876 | 0.905 | 0.885 [0.851, 0.917] | −0.056 [−0.090, −0.021] |

All random packs, paired minus v2 per stratum:

| stratum | n | qwen_v3 − v2 | recut_v3 − v2 |
|---|---|---|---|
| U | 509 | −0.100 [−0.130, −0.071] | −0.096 [−0.126, −0.069] |
| D | 300 | +0.110 [+0.053, +0.163] | +0.080 [+0.030, +0.130] |

Per column, all random packs. The first three columns count cells right; the last is qwen − v2, stratum-weighted.

| column | n | v2 | qwen | recut | qwen − v2 |
|---|---|---|---|---|---|
| ACL | 55 | 47 | 54 | 53 | +0.165 |
| Lateral Meniscus | 46 | 40 | 45 | 44 | +0.192 |
| Medial Meniscus | 52 | 43 | 51 | 47 | +0.164 |
| Contusion | 41 | 33 | 37 | 37 | +0.068 |
| PF OA | 48 | 42 | 44 | 42 | +0.040 |
| Medial OA | 69 | 51 | 58 | 55 | −0.005 |
| Fracture | 120 | 119 | 118 | 117 | −0.006 |
| Baker's | 43 | 43 | 40 | 38 | −0.080 |
| Synovitis | 27 | 26 | 26 | 25 | −0.084 |
| Lateral OA | 133 | 120 | 112 | 118 | −0.095 |
| MCL | 125 | 124 | 101 | 104 | −0.178 |
| **Effusion** | 50 | 46 | 30 | 29 | **−0.241** |

Per-column intervals are wide (n 27–133), so each column is directional. The tear/contusion block is the only
grouping named here, and it matches the mechanism knee-clzzs names (v2 reads degeneration as tear).

## Reading

1. **No blind ruler shows the learned labeler beating v2 pooled.** It is −0.055, and the interval excludes zero.
   This reproduces the knee-clzzs note C numbers (v2 0.941, qwen 0.886) with intervals. The gold-58 +0.028 does
   not survive a ruler the tables never saw.
2. **The loss is not uniform. It is a criterion line.** Effusion (−16 cells) and MCL (−23 cells) carry most of it.
   Those are exactly the columns where the criteria-era reader applies a severity threshold ("moderate or
   large", grade). Suspect the test before the idea (faulty TEST): this ruler's rubric is **UNVERIFIED** for
   root402, because the pack stores no rubric. The holdout469 rubric was "affirmed at **any severity**". If
   root402 used the same rubric, a labeler that applies the host's severity criterion is marked wrong by
   construction. Gold-58, which is radiologist image truth, favours v3 in the same place. So the pooled −0.055
   is a verdict on "any-severity mention", not on the competition's label definition.
3. **The tear block is a clean win with no criterion confound.** ACL, both menisci and Contusion gain +0.07 to
   +0.19 per column, on cells where "tear present" has no severity threshold. On root402, a post-hoc hybrid
   (qwen on those four columns, v2 elsewhere) reads **+0.015 [−0.010, +0.039]**. It was chosen by looking at this
   ruler, so it is selection-optimistic and not a number to plan on.

## Verdict on the 4 h arm

**A whole-table v3 training arm is NOT licensed.** Its pooled label accuracy is below v2 on the only blind
ruler, and the one ruler where it wins is circular.

Two cheaper, better-posed steps come first:

- **Effusion/MCL criterion question (zero card).** Re-adjudicate ~40 random Effusion and MCL cells blind under
  the host criterion rubric. If v3 wins there, the "loss" was rubric mismatch and a whole-table arm re-opens.
- **Tear-column hybrid (instrument only).** A table with Qwen on ACL, Medial Meniscus, Lateral Meniscus and
  Contusion. Its label-accuracy gain is +0.015 with an interval covering zero, and it was selected on this
  ruler. Validate it on a fresh blind tear-column pack (~60 cells, zero card) before any card hour.

Script: session scratchpad `blind_ruler.py`, seed 20260915. Per-cell CSV not committed (data out of repo).
