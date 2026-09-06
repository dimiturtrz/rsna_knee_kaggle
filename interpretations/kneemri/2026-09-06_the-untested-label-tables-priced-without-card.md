# The six untested label tables, priced on CPU before any of them got an arm

*2026-09-06, while `chaindense` held the card.*

## Why this was worth ten minutes

`blend_three_v1.csv` is the target table behind the biggest single recipe term this project owns
(+0.0073 at the 224 corner, and one of the three links in tonight's +0.0089 chain). The labels
directory holds twenty tables. **Six of them have never appeared in a single arm** — `blend_four`,
`blend_hybrid_fx_v4`, `blend_hybrid_fx_v4b`, `blend_hybrid_fxneg`, `blend_hybrid_fxrub`,
`blend_nolixin`, plus `report_labels_v4` and the duplicate `confident_v2`. Each is a plausible
70-minute arm. That is seven hours of card sitting in a directory, and none of it had a price.

## The distance map, every table against `blend_three_v1`

Mean absolute difference over 4,406 studies x 12 findings, Pearson over the same cells, and the
count of cells whose `__verdict` disagrees (52,872 cells total).

| table | mean\|d\| | corr | verdict flips | arms |
|---|---|---|---|---|
| `blend_four` | 0.0114 | 0.99915 | **0** | 0 |
| `blend_nolixin` | 0.0364 | 0.99221 | 0 | 0 |
| `blend_hybrid` | 0.0524 | 0.98718 | 0 | 2 |
| `blend_hybrid_fx` | 0.0535 | 0.98616 | 0 | 1 |
| `blend_hybrid_fxrub` | 0.0573 | 0.97360 | 0 | 0 |
| `blend_hybrid_fx_v4b` | 0.0585 | 0.97186 | **508** | 0 |
| `blend_hybrid_fx_v4` | 0.0586 | 0.97166 | **514** | 0 |
| `blend_hybrid_fxneg` | 0.0600 | 0.96604 | 0 | 0 |
| `confident` / `confident_v2` | 0.0879 | 0.95772 | 0 | 12 / 0 |
| `report_labels_v3` | 0.0931 | 0.91391 | 0 | 2 |
| `report_labels_v2` (the base) | 0.0950 | 0.95046 | 0 | many |
| `flatyes_v2` | 0.0962 | 0.94123 | 0 | 1 |
| `report_labels_v4` | 0.0989 | 0.93678 | **514** | 0 |

**Four of the twenty are byte-equivalent duplicates of tables already in use** and can never be an
experiment: `merged_blend_v1` = `blend_three_v1` (distance exactly 0.0000), `merged_labels` =
`merged3_labels` = `report_labels_v2`, `merged_tagged_labels` = `merged3_tagged_labels` =
`report_labels_v3`, `confident_v2` = `confident`. An arm on any of them would have re-run a known
result under a new filename, which is the failure mode the folds-and-seeds ban exists to stop.

## `blend_four` is dead, and it did not need a card to die

A fourth reader moves the targets by **0.0114 mean absolute** and flips **zero** of 52,872 verdicts.
The step that actually bought something — `report_labels_v2` to `blend_three_v1` — is **0.0950**, i.e.
**8.3x larger**, and it bought +0.0073. If the supervision effect is anything like monotone in how far
the table moves, a fourth reader is worth **+0.0009**: a seventh of the 0.0062 paired floor, for
seventy minutes of card. It is not a close call and it is not worth a confirmation seed.

This is also the third reader's own story told again. `2026-08-26_the-label-table-axis-is-exhausted.md`
measured `hybrid_v4 − blend_four` = +0.0082 [+0.0005, +0.0183] on the 58 gold studies — the blend with
four readers is *separated below* the single best reader. Adding readers dilutes.

## The v4 family is the only real question here, and it prices at ~+0.002

Only three tables flip verdicts, and they are the same edit: `report_labels_v4`,
`blend_hybrid_fx_v4`, `blend_hybrid_fx_v4b`, all carrying ~510 flipped cells. Everything else is
smoothing. A flipped verdict is a **correction**, a different kind of change from a softened target,
and it is the one mechanism in this directory that the +0.0073 blend term does not already contain.

It is still not worth an arm, and the existing rulers say so:

- On the 58-study gold ruler, `hybrid_v4 − blend_three` = **+0.0079 [−0.0008, +0.0180]** — the best
  single reader, a hair off separation from the blend we already train on.
- The ruler-to-supervision transfer is already measured once: `blend_three − ours` = +0.0254 on that
  ruler and **+0.0073** as supervision at the arm level, a ratio of **0.29**.
- Applying it: +0.0079 x 0.29 = **+0.0023**. Under the 0.0062 floor, on a 70-minute arm.

**PRICE, said out loud:** ~70 min of card, plausible gain +0.002, P(clears the 1.5% bar) ≈ 0. That is
worse than the arm currently on the card, which is pre-registered to test something worth +0.015.

## What this does and does not close

**Closes:** `blend_four` as supervision, on arithmetic rather than on a run. The four duplicate
filenames, permanently — they cannot be experiments. And the *default* assumption that an unused
table in the directory is an untested idea; most of them are the same idea at a different smoothing
strength.

**Does not close:** the v4 verdict corrections. They are priced at +0.002, not refuted, and the price
rests on a transfer ratio measured **once**, on one pair of tables. If a cheap ride ever appears — a
run that is happening anyway and can take `blend_hybrid_fx_v4b` instead of `blend_three_v1` at no
extra cost — take it, because a correction axis is mechanically different from a smoothing axis and
the 0.29 ratio was fitted on smoothing. What it must not get is a dedicated arm ahead of anything
targeting +0.015.

**The habit:** twenty files, six of them plausible arms, seven hours of card implied — and the whole
set was ranked in two CPU measurements against tables whose effect was already on record. The distance
map is what turned "six untested ideas" into "one idea at six smoothing strengths, priced at a third
of the floor".
