# A seat is earned by what a member ADDS, not by what it scores alone

*2026-08-16 · cached fold-0 holdout, 909 studies × 12 labels, no GPU, no submission slot spent.*

Three questions were priced offline against the cached member predictions
(`holdout_predictions/*.npy`). Only one of them bought anything.

## 1. Subset selection: +0.0031 over the shipped entry, +0.0024 over `strongest`

`MemberPackage.strongest(n)` sorts members by their own solo holdout. That is a MARGINAL ranking, and an
ensemble is not marginal. Over eight cached members:

| subset | holdout AUC |
|---|---|
| `strongest(2)` = res392 + calibrated — the subset the LB-0.919 entry ran | 0.8581 |
| `strongest(4)` | 0.8588 |
| **named best-4** = res392 + intensity392 + calibrated + sliceattn | **0.8612** |
| named best-4 + extended_ws | 0.8610 |
| all six | 0.8603 |

The solo sort keeps `extended_ws` (solo 0.8462) and drops `intensity392_ws` (solo 0.8459) — a 0.0003
difference in the marginal that costs 0.0024 jointly. Hence `MemberPackage.named()` and `KernelRun(names=…)`:
an entry now NAMES its members, so the number computed offline is the number the kernel runs. A missing name
is fatal rather than skipped — a silently smaller ensemble would ship a different entry than the one whose
score justified the slot.

+0.0031 is below the 0.0053 noise floor and far below the 1.5% bar. It is directional, and it is FREE
(same weights, same decode, same runtime), which is the only reason it is kept.

## 2. Per-label combining: worth nothing here. This closes an open question.

`2026-08-14_the-ruler-was-five-times-too-small.md:104` left open whether picking a different member subset
PER LABEL beats one flat rank-average. It does not, for diverse members:

- split-half peak (top-5 of 6 per label) 0.8603 vs flat rank-average 0.8590 → +0.0013, a quarter of the floor
- the per-label ORACLE — the selection that reads the answers — scores 0.8590 against the plain average's
  0.8588. Total headroom **+0.0002**.

An oracle that cannot beat the average by more than 0.0002 means there is no per-label structure left to
exploit; every honest heuristic is bounded by it. The question is settled, not deferred.

## 3. Rotation TTA: priced, not bought

+0.0004 (calibrated) / +0.0010 (sliceattn) / +0.0020 (lora_v3) for 3× the inference cost. On the accuracy
board it is under the floor; on the efficiency board it is strictly negative.

## What the entries are

- `kneemri-chosen` — the named best-4, disjoint windows. 0.8612 offline against the 0.8581 of the subset
  behind LB 0.919.
- `kneemri-onedecode` — calibrated + sliceattn + extended_ws, all one pixel group. 0.8553 offline: 0.0059
  of holdout traded for HALF the decoding. Whether that trade is worth it is the one thing the holdout
  cannot say, because the holdout does not carry a clock — only the efficiency board can price it.
