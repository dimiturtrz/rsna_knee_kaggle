# Our per-column offline AUC ranks our LABELS, not our model — and the board headroom is in the ten columns we have never read

**Date:** 2026-09-05 · **Card spent:** none · **Slots spent:** none (reads two that were already bought) · **Board read:** 2026-09-05T11:08:57Z

`colnull` fixed the competition metric as a plain unweighted mean of twelve per-column ROC AUCs, which
makes a damaged-column entry a measuring instrument: overwrite one column with a constant 0.5, and the
board drop gives that column's board AUC as `(champion − damaged) × 12 + 0.5`. Two such entries have now
scored against the same 0.944 champion control.

| column | our offline AUC | board AUC | lift |
|---|---:|---:|---:|
| MCL (`colmcl_v1`, ref 56020054, board 0.904) | 0.8104 — our **worst** of twelve | **0.980** | **+0.17** |
| Baker's (`colbaker_v1`, ref 56023244, board 0.903) | 0.9208 — our **best** of twelve | **0.992** | **+0.071** |

**The rounding, stated before the reading.** Board scores carry three decimals, so each is ±0.0005 and a
champion-minus-damaged difference is ±0.001, which the ×12 multiplies to **±0.012 on a column AUC**. The
two column values are therefore 0.980 ± 0.012 and 0.992 ± 0.012: they overlap, and **the ordering between
these two columns is NOT resolved.** Everything below rests on the part that survives that interval —
both columns sit at or above ~0.97 on the board, and both sit far above their offline values.

## What the two points say

**1. The level does not transfer, and this is now measured twice rather than argued once.** Our
champion pool averages 0.8715 across twelve columns offline; the board averages 0.944. Both externally
read columns land higher on the board than offline, by +0.17 and +0.071.

**2. The lift is not uniform, and that is the whole finding.** A constant offset would be a boring ruler
calibration. Instead the column where we look *worst* lifts by 0.17 and the column where we look *best*
lifts by 0.071. MCL is our weakest column offline and is at 0.980 on the board — the model separates MCL
nearly perfectly against the competition's labels while scoring 0.8104 against ours.

**So the offline column ranking is substantially a ranking of OUR SUPERVISION.** MCL has 50% radiologist
agreement, the second-worst-labelled column this project owns, and `overcall_audit` measures our lexicon
at 108 false positives to 32 misses. A model evaluated against noisy labels is attenuated toward 0.5, and
attenuated most where the noise is worst. That is exactly the shape of these two points.

**3. Therefore the per-label lane is dead as a planning tool, and for a better reason than before.** The
+0.0196 bottom-three headroom in the per-column map is priced on the assumption that a column we score
badly is a column with board headroom. On the one column where that assumption has an external check, it
is wrong by 0.17: MCL is nearly solved on the board. "Fix our weak columns by modelling" aims at label
noise wearing a model's clothes.

## Where the headroom actually is

The board mean is 0.944, so the twelve columns sum to 11.328. Two of them are ~0.980 and ~0.992, which
leaves **the other ten averaging ~0.936** — below both measured columns, and we have no external read on
any of them. That is where the 0.010 gap to first place lives, and nothing offline can point at it,
because the offline map has now failed its only two external checks in LEVEL and cannot be trusted in
ordering either.

**This is a case for spending slots on instruments rather than entries.** A damaged-column entry costs
one slot and returns one column. A damaged-GROUP entry costs one slot and returns the SUM over the group,
which localises faster: knock out six, and a low sum says the weak columns are inside that six. Three
slots spent adaptively would narrow twelve unknown columns to a short list, and every seating and
supervision decision after that would be aimed instead of guessed.

## What must NOT be concluded

- **Not that the board labels are clean.** They may be noisy too; nothing here measures that.
- **Not that case mix is ruled out.** `Q-RULER5` established that case mix and a definitional mismatch
  between our lexicon's MCL and the competition's make *identical* offline predictions and cannot be
  separated without board labels or board cases. Label noise is the leading explanation with measured
  support, not the proven one. A hidden test with different prevalence would also lift AUC.
- **Not that paired member comparisons are invalid.** A shared bias plausibly cancels when two members
  are compared on the same holdout, which is the only reason the eight declines and seven board flats
  still stand. That instinct remains **unverified** (knee-l4xx6), and it is weakest exactly where members
  differ in SUPERVISION — the `errors_*` family.
- **Not that the two columns are ordered.** ±0.012 covers the 0.012 between them.

Beads: knee-zsyp4, knee-l4xx6, knee-aily5, knee-wko44.
