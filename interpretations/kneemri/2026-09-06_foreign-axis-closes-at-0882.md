# The foreign family reads 0.882 on the hidden split, and its own pre-registration closes the axis

2026-09-06, 23:00Z. `foreignsolo_v1` (ref 56063670) put mattiaangeli's five DINOv3 fold checkpoints on
the board **alone** — 5/5 folds mounted, rank-mean, **zero members of ours** — and scored **0.882**.

## What the number is, and what it is not

It is the **first honest number any foreign checkpoint has ever had here.** Every previous foreign
reading was in-sample: `foreign_overlap` shows their published label table covers 4349 of our 4407
studies, so 896 of our 909 fold-0 holdout studies are inside it. No offline ruler we own can price these
weights. The board can, and did.

It is **not** a shortfall against a published number. The submission message and this row's first
version priced 0.882 against "their published solo 0.939". **There is no such number.** Nothing in
`research/`, `interpretations/` or `comp/` carries a board-comparable solo for these checkpoints, and
`2026-09-06_foreign-folds-three-retractions.md` says outright that they had never been priced on our
holdout at all. The 0.939 is **our own** six-member score, lifted out of prose into a pre-registration —
the same defect that mis-quoted `board_transfer`'s slope twice. `comp/board.jsonl` carries the retraction
on the row itself.

## The pre-registration decides it, and it was written before the slot was spent

`comp/kernels/kneemri-foreignsolo/kneemri-foreignsolo.py:33-38`:

- `S >= 0.940` — family is board-strong, the -0.013 is the blend, re-run the seat at low weight.
- `0.920 <= S < 0.940` — a diversity seat only, weight near 1/10, no further slot this cycle.
- `S < 0.920` — **the family does not transfer to the hidden split, the foreign axis closes for THIS
  family**, and a second family needs its own board reading before any card time is spent on it.

S = 0.882. Third band, by 0.038 — thirty-four times the 0.0011 board resolution. The axis closes.

## The harness was checked before the verdict, and it is faithful

The serving path is not a re-implementation. `ForeignEntry` (`kneemri/kaggle/foreign_entry.py:35,93-98`)
runs `ForeignFolds`, which lifts **their own source** and builds **their own classes**
(`foreign_folds.py:151-157`): `timm.create_model` with their `backbone`/`img`/`in_chans`, then
`ViTSlotToken(encoder, N_SLOT_TYPES)` on the `cond == "token"` path — the slot-token conditioning is
present, not dropped — then their `Net` with their `pool`. `load_state_dict(..., strict=False)` is
followed by a refusal: any missing key outside `enc.` or any unexpected key raises rather than degrading
quietly (`:157-160`). The fold key is transcribed from their `md5(report)[:8] % 5`, not inferred.

So the two hypotheses that a serving audit would have separated are already separated by construction.
There is no silent wrong-function to find; the family is weaker than ours on this test.

## What this costs, what it buys, and what it does NOT rule out

**Costs:** the foreign-family seat was the last axis still reading above resolution offline (+0.0178 at
6x). That reading was in-sample and is now void as a seat argument.

**Buys:** `foreignseat_v1`'s -0.013 is explained without confounding. It is the family, not the
`LogitAverage -> RankAverage` swap — the rank rule is exonerated as the carrier, which it was under
suspicion for.

**Does NOT rule out**, and this is the part a kill-word would have thrown away:

1. **Distillation FROM their predictions.** A teacher does not need to be board-strong to be
   informative; `foreign_distill_*` arms price a soft target, not a seat. 0.882 says nothing about
   whether their errors are decorrelated from ours where it matters.
2. **A different foreign family.** The band names this explicitly: a second family needs its own board
   reading, not an inherited verdict.
3. **Their split key**, which remains public, transcribed and useful for honest fold-matched pricing of
   anything of theirs.

What it does close: any slot, and any card hour, spent seating THIS family.
