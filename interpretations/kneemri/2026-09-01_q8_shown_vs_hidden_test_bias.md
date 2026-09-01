# Q8 (knee-5nbv): bias between the 30% shown test and the 70% hidden test

2026-09-01, Kimi. Read-only audit against `comp/board.jsonl` and the transfer tools already in the repo.

## What can be measured

The only thing on disk is the **offline-to-public-board** relationship. `kneemri/evaluation/board_transfer` reports:

```text
33 entries on the ensemble scope, 414 orderable pairs
compression: median +1.23 board points per holdout point over the WHOLE record
  among entries at or above 0.8670: 16 resolved pairs, median +0.98 [+0.94, +1.08]
  among entries at or above 0.8692:  4 resolved pairs, median +0.80 [+0.70, +1.19]
inversions: 9/414 pairs the board ordered the other way, 0/264 of the pairs the holdout resolved at all
```

So the compression is **not a constant**: it falls from +1.23 over the whole record to +0.80 in the champion regime. The holdout has **never mis-ordered a resolved pair** on the public board. The 9 inversions all sit under the holdout's own noise floor (offline delta < 0.0053), so they are cases where the ruler called the entries equal and the board broke the tie.

## What cannot be measured

The **public/private 30/70 split** is unknowable until the competition closes. `board.jsonl` records Kaggle's public score, which is a scalar on 30% of the hidden set. Any claim about whether that 30% is biased relative to the remaining 70% — by label prevalence, by site, by acquisition — is not testable from the repo. The only honest statement is that the public score is the only board signal we have, and the transfer tools treat it as ground truth for ranking.

## Residuals: do they correlate with anything we control?

I re-read the transfer among the 15 strongest entries (offline ≥ 0.8670) using the local median ratio +0.978 and computed each entry's mean pair residual (actual board delta minus predicted delta).

| control | correlation with residual | n | comment |
|---|---|---|---|
| member count | r = +0.122 | 15 | no correlation |
| offline score | r = +0.037 | 15 | no correlation (saturation is in the ratio, not a per-entry residual) |
| date index | r = +0.222 | 15 | weak, not significant |
| `selection=search` | mean residual +0.0014 (1 entry) vs −0.0003 for the rest | 15 | one point, no conclusion |
| fold -1 share | not computable | 0 | most rows carry no `members_named`; the ones that do have zero `_ff` members in this subset |

The only visually large residuals are single entries, not systematic:

- `crossenc_v1` (board 0.926, offline 0.8678) sits −0.0039 below the local line.
- `peak8_v1` (board 0.932, offline 0.8731) sits −0.0030 below, and it is the one search-selected row that was corrected for selection bias — the raw 0.8731 was already reduced.

None of these are large enough relative to the board's three-decimal precision and the pair-to-pair spread to claim a controllable bias.

## Rule for spending a slot

Use the **local** compression, not the whole-record median. At the champion regime it is about **+0.80 board points per holdout point** with a bootstrap interval of roughly [+0.70, +1.19]. A slot is justified when:

1. The offline gain is **resolved by the holdout**: offline delta ≥ 0.0053 (the ruler's own floor).
2. The predicted board gain (`0.80 × offline_delta`) exceeds the current board gap to the entry being beaten.
3. The improvement is **pre-registered**, so a flat or negative result is readable as a closure rather than noise.

At the current champion offline (~0.8690, board 0.931), an offline gain of +0.0062 (one noise floor) predicts a board gain of about +0.0050. That is the smallest gain worth a confirmation slot; anything smaller is inside the transfer spread and the inversion risk.

## Verdict

- **Measurable**: offline-to-public-board compression is a curve that saturates at the top; the holdout has not mis-ordered a resolved pair on the public board.
- **Unmeasurable**: any claim about the 70% hidden test population is speculation until the competition ends.
- **Controllable bias**: none visible in member count, date, offline score, or selection method. Fold -1 share could not be checked because the strongest rows do not name their members.
