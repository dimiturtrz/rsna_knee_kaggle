# The ruler had four rows outside it, and the one that mattered is the only non-DINOv2 point

**Date:** 2026-09-04 · **Beads:** knee-q18e8 · **Status:** measured, with a confound that is total

## The defect

`comp/board.jsonl` carries a `scope` field that says which DENOMINATOR a row's `offline` number was
computed on. Its vocabulary is `ensemble` (the holdout AUC of the thing submitted), `member_mean`
(the mean of the members, comparable only to other member means) and `excluded` (deliberately not
priced, reason recorded).

Four rows carried `scope: "included"`. That is not a value in that vocabulary — it is the negation of
`board_check`'s `EXCLUDED` sentinel, written into a field that means something else. Nothing caught
it, because the two consumers read the field differently and both were satisfied:

- `board_check` (`board_check.py:296,335`) only ever tests `scope != EXCLUDED`, so `"included"` passes
  every check it makes.
- `board_transfer` (`board_transfer.py:201`) fits `row.get("scope") == self.scope`, so `"included"`
  matched nothing and the rows dropped out of the fit — counted, but as an unnamed non-comparable
  scope, which reads as intended rather than as a defect.

The rows were `fiveship_v1`, `holdfive_v1`, `ortho_solo_v1` and `foldfour_v1`. Three carry a real
ensemble-scope offline number and an `offline_scope` prose line stating the 909 shared fold-0 holdout
studies they were computed on. `fiveship_v1`'s offline is null and it drops out of the fit anyway, by
the mechanism designed for exactly that (`_carries_both`).

All four are now `ensemble`.

## What the ruler does with them

| | before | after |
|---|---|---|
| entries on the ensemble scope | 34 | **37** |
| orderable pairs | 436 | **530** |
| compression, whole record | +1.25 | **+1.19** |
| ≥0.8678 | +0.97 [+0.94, +1.17] | **+0.97 [+0.94, +1.17]** (13 resolved pairs) |
| top cell | +0.71 [+0.70, +0.89] (≥0.8694) | **+0.80 [+0.71, +0.89]** (≥0.8695, 2 pairs) |
| inversions the holdout RESOLVED | **0 / 276** | **2 / 342** |

The planning constant does not move — +0.97 at the operating point is unchanged, so nothing priced
against it is re-priced. The chain in `2026-09-03_the-0953-chain-is-short.md` stands as written.

**The inversions are the finding.** The record's headline property for three weeks was that the
holdout never once ordered a resolved pair the wrong way round. It now does, twice, and both
inversions have the same entry on one side:

```
solo_v1     over ortho_solo_v1:  holdout +0.0066,  board -0.039
onedecode_v2 over ortho_solo_v1: holdout +0.0101,  board -0.003
```

That is ONE observation, not two — one entry against two partners.

## What it says, and the confound that stops it saying more

`ortho_solo_v1` is `ortho_dense_s2026` served alone: DINOv3 ViT-L continued on 1.25M knee slices, the
only non-DINOv2 board point this project owns. It carries the LOWEST offline of all 37 entries
(0.8452) and reads 0.907 — level with `weakseven_v1` (0.8562 → 0.908), which is 0.011 better offline,
and 0.039 above `solo_v1` (0.8518 → 0.868), which is 0.0066 better offline.

Against the whole-record slope its offline predicts roughly 0.886. It read 0.907. **About +0.02 board
above where the DINOv2-fitted ruler puts it.**

**The confound is total and must not be argued around.** Offline level and date are collinear in this
record: every low-offline entry is from 2026-08-12..08-24 and every late entry is high-offline. There
is no late-dated low-offline entry anywhere in the 37. So "the ortholarge trunk transfers better than
its holdout says" and "the current kit transfers better than the August kits did" predict the same
observation here, and these data cannot separate them. A same-era line (08-16..08-20, 0.8553 → 0.8678)
has slope +1.76 and still leaves ortho_solo +0.021 high, but any kit-era term pushes a 09-03 entry's
prediction up, so part of that residual is era by construction.

This is precisely the collinearity the P1 bead filed on 2026-09-03 proposes to break: two
**current-kit** DINOv2 solos served alone at `--fold 0` at separated offlines. Candidates with honest
fold-0 records: `calib_dense_hold_s2026` 0.8634, `small_batch6_s2026` 0.8598, `through_dense_s2026`
0.8558, `calib_span_s2026` 0.8545 (the outer two span 0.0089, above the 0.0053 resolution floor).
Cost: two slots, zero card. The bead priced that instrument as speculative. It is not speculative any
more — there is a candidate effect worth +0.02 board on the table, and the instrument is the only
thing that can tell whether it belongs to the trunk or to the kit.

## The branch this changes, before the arm lands

`ortho_stack` is on the card now, and its middle pre-registered branch reads: within the 0.0062 floor
of `band_dense` = 0.8578 means the continuation gain is resolution-bound, **and the in-domain trunk
lane closes**.

That branch is written on the assumption that offline order transfers to the board. For this trunk
family that assumption is now the thing in question. If ortholarge really sits ~+0.02 above the
DINOv2 line, an `ortho_stack` that merely MATCHES `band_dense` offline is not a closed lane — it is a
member that ties the incumbent on our ruler and beats it on the board.

**The branch is not withdrawn and its number is not moved** — moving a pre-registered threshold
after seeing a related measurement is how a result becomes an artefact. It is annotated: a middle
reading closes the lane *conditionally on the two-solo instrument coming back null*. If that
instrument says the residual is kit-era, the branch stands exactly as written.

## What this does not say

It does not say the holdout is broken. 342 resolved pairs, 2 inversions, both the same entry: the
ruler is sound everywhere it has DINOv2 points on both sides, which is everywhere except here. It
does not license reading any offline number through a wider slope — the +0.97 at the operating point
is unchanged and is still the planning constant. And it does not make `ortho_dense_s2026` shippable;
it is declined in writing at `comp/members.declines:178` on a seat argument this measurement does not
touch.
