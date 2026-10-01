# The ruler map: Effusion was ranking small fluid

2026-09-07, 12:40–13:30Z. CPU, read-only, researcher lane. Four letters (D22–D25) over the same members, per
column, asking which ruler is allowed to answer which question. Numbers from `scratchpad/d22_ruler.py`,
`d23_ruler3.py`, `d24_ladder.py`, `d25_recut.py` (`d25_recut.out`), `d25_colseat.py` (`d25_colseat_novoid.out`)
and `d26_contusion.py` (`d26_contusion.out`); the letters in `inbox/mastermind/read/` carry the full tables.

## The three rulers

| ruler | rows | truth | what it is |
|---|---|---|---|
| **gold-58** | the 58 `train.csv` rows with all twelve labels (n+ per column 9–35; 57 of them in v2) | radiologist, image-read | the only pixel ruler; expert-annotated like the hidden test |
| **LEDGER-896** | each member's `<name>.holdout.npz`, 909 studies minus the 13 gold rows | `report_labels_v2` soft value binarised at the midpoint, `targets > 0.5` (`kneemri/evaluation/scores.py:19`) | our own text reader, binarised |
| **HIS** | `D:/data/volumetric/mri/processed/raptor/labels/labels_llm_soft.csv`, 4,349 studies, all 896 present, 0 of the 58 | a second text reader of the same reports; 14 values with a gap 0.45–0.76, so >0.5 ≡ ≥0.75 | provenance: nothing on disk beyond the filename (D24 job 3) |

Members: the 30 with both a 58-row reading cache and a holdout npz; 10 of them hold the 58 out honestly.
Agreement between two rulers = Spearman across the 30 members of the per-column AUC. v2's soft values take
five levels only — 0.08 / 0.28 / 0.68 / 0.82 / 0.94 — so every cut between 0.28 and 0.68 is the same cut
(the midpoint), and every cut between 0.68 and 0.82 is the same cut (0.75).

## D22 — the map at the midpoint

```
column     LEDGER~gold   HIS~LEDGER   HIS~gold    verdict                          gold paired ±   LEDGER paired ±
ACL          +.61          +.76         +.60      PASS — three rulers agree            .026            .016
MCL          +.69          +.84         +.76      PASS                                 .065            .023
MedMen       +.46          +.50         +.09      PASS (gold weak)                     .032            .010
LatMen       +.48          +.54         +.66      PASS                                 .065            .020
MedOA        +.35          +.83         +.37      MARGINAL — text rulers agree         .042            .011
LatOA        +.37          +.47         +.34      MARGINAL                             .071            .017
PFOA         +.49          +.87         +.59      PASS                                 .048            .016
Effusion     −.17          −.17         +.78      FAIL — LEDGER is the odd one out     .085            .013
Synovitis    −.37          −.49         +.46      FAIL — LEDGER is the odd one out     .083            .017
Bakers       +.47          +.02         +.07      PASS on LEDGER~gold; HIS scatter     .037            .014
Contusion    −.15          +.68         −.23      FAIL — gold is the odd one out       .056            .014
Fracture     −.20          +.76         −.07      FAIL — gold is the odd one out       .064            .029
macro        +.59          +.70         +.79
```

Paired ± = median 95% half-width of a member-vs-member column AUC difference, 2,000 study bootstraps.
Hold-only n=10: Fracture LEDGER~gold −.79, HIS~LEDGER +.86 — the same map, sharper.

## D23 — the third reader says which side is wrong

A two-ruler disagreement cannot say which ruler is wrong; a third one can. On Effusion and Synovitis
HIS sides with gold against LEDGER (HIS~gold +.78 / +.46, HIS~LEDGER −.17 / −.49): the text CAN carry the
radiologist's ordering, and our binarisation of it does not. On Contusion and Fracture the two text
readers agree with each other (+.68 / +.76) and the 58 disagrees with both: gold n+ = 19 / 18, half-widths
.056 / .064, and the 58's Fracture prevalence is .310 against .075 on the corpus — the 58 is not sampling
the corpus's fractures. Bakers: LEDGER~gold +.47 is the ruler; HIS's ordering is scatter (its own column
AUC sits at .960, every member at the ceiling).

## D24 — why Effusion fails: the 0.68 rung

Of the 1,510 corpus studies v2 marks positive and HIS negative (34.7% of 4,349), 1,463 (96.9%) sit on
v2's 0.68 "possible" rung. Forty seeded quotes (`FindingMentions`, Effusion vocabulary at
`core/data/labels/finding_mentions.py:119-128`) read "small / mild / minor / minimal / trace" in nine
languages — 32 of 40 size-graded small, 0 negation misreads, language mix proportional to the corpus. HIS
files small effusion at 0.20. The two readers read the same thing; the midpoint cut turns the rung positive.
Rung × HIS on the 896: 0.68 → 294 HIS− / 23 HIS+; 0.82 → 10 / 187; 0.94 → 0 / 34.

Against gold (57 of 58 in v2) the 0.68 rung splits 14 negative / 13 positive: the radiologist counts small
fluid half the time, so the rung is genuinely ambiguous. Cutting at 0.75 moves cell agreement with gold
.719 → .737, one cell — what the cut buys is not agreement, it is the ORDERING: the ledger stops rewarding a
member for ranking small fluid as high as moderate fluid, which is the ordering both other rulers follow.
The training target for a 0.68 cell is already near the fence (pooled 0.378 at
`core/data/labels/targets.py:40`, plus Effusion's +0.432 shift at `:54`); the ruler was not.

**Synovitis is a different mechanism.** Of the 719 corpus studies HIS marks positive and v2 negative, 710
are v2 UNK (0.28) — level, not lexicon, and below every cut. What HIS keys on is effusion size:
P(HIS Syn+ | HIS Eff+) = .924, P(HIS Syn+ | HIS Eff−) = .011; by v2 Effusion rung the HIS Synovitis-positive
rate runs .005 / .019 / .081 / .874 / .941 at 0.08 / 0.28 / 0.68 / 0.82 / 0.94. Only 11.7% of the 719
mention a synovium word. HIS has two Synovitis positives — 0.80 where the report says synovitis (v2 YES
318 of 322) and 0.76 where the report says moderate-or-larger effusion (v2 UNK 587 of 608). Gold finds
synovitis in 72% of 0.82-effusion knees, so the proxy has image truth in it; it is still a proxy. A
Synovitis head trained with HIS learns "moderate effusion" twice and the HIS Synovitis ruler reads it high
by construction; gold is the only Synovitis ruler outside that circle.

## D25 — the recut: `LabelScores(cut=0.75)` repairs Effusion and nothing else

Same 30 members, truth = the v2 table, the 0.68 rung made negative (`scores.py:90`, `targets > cut`). This
is the pre-registered cut (`scores.py:27-39` argues it), not one picked for the Spearman: the sweep
0.5 / 0.75 / 0.9 was run and 0.9 reads Effusion +.79 all / +.48 hold against 0.75's +.74 / +.77.
LEDGER~gold per column at 0.5 → 0.75, positives lost on the 896 in brackets:

```
ACL  +.61→+.69 (40%)   MCL  +.69→+.77 (67%)   MedMen +.46→+.12 (25%)   LatMen +.48→+.68 (39%)
MedOA +.35→+.19 (31%)  LatOA +.37→+.38 (45%)  PFOA   +.49→+.62 (24%)   Bakers +.47→+.04 (49%)
Effusion −.17→+.74 (58%)   [hold-10: −.16→+.77]
Synovitis −.37→+.05        Contusion −.15→−.38        Fracture −.20→+.13 [hold −.79→−.75]
macro ordering vs gold: +.59 → +.80  [hold +.25 → +.71];  0.75 on the four blind columns only: +.77 [+.60]
```

The eight PASS columns are the control: mean |move| .18 against .47 on the four, and two of them (MedMen,
Bakers) LOSE their ruler at 0.75 because their 0.68 rungs carry real positives (20% / 50%, n 5 / 6).

Why only Effusion: it is the one column whose coin-flip rung sits ABOVE the midpoint and below 0.75 (v2
0.68 = 48% gold-positive, n 27 of 57). Synovitis's mass is UNK 0.28 (41 of 57), below every cut, and its
0.68 rung is 78% positive — the recut is flat there (+.05, p .79). Contusion's coin-flip rung is 0.82
(46% positive, n 24), above every cut, so every recut is worse (−.38). Fracture has no coin-flip rung and
keeps its hold-10 sign under every cut. A per-column search over six cut variants (48 choices on 58 rows)
gave macro +.79 against the uniform +.80 — nothing; the uniform second column is the one to carry.

Effusion ordering under the recut: gold's top five (`stackfit`, `contrast_dense`, `base_clean`,
`labelwise_dense`, `warm_calib_lowlr`) all sit in the ledger's top seven; mean |gold rank − ledger rank|
10.8 → 4.9 of 30. The relabel-vs-band "Effusion inversion" (D22 §4) was the reference case:
`band_dense_hold` is 29th of 30 on gold Effusion and joint-first on midpoint-LEDGER Effusion; at 0.75 it
is 27th.

## The seat map under LEDGER75 (`d25_colseat.py`, 255 honest npz members over the 896 + `raptor_v4_swa`)

`dinov3f0_solo` is EXCLUDED: it served released foreign fold weights, 703 of 909 in-sample
(`experiments/README.md:8860`, `comp/members.declines:694`); the D25 letter counted it honest and its
"seats" are void, which changes no conclusion below.

- Midpoint (R1): two members clear +0.02 over the five-parent core on any column — `foreign_oof_corpus`
  on MCL / LatOA / Effusion / Synovitis / Fracture and `merged_blend_cf_r392` on Fracture. Champion proxy
  minus best honest solo: Effusion −.035, Synovitis −.033.
- LEDGER75 (R6): the Effusion and Synovitis seats vanish; six members clear +0.02 — all on Fracture
  (`foreign_oof_corpus`, `merged_blend_cf_r392`, `raptor_ft_repl2x2_a`, `_swa`) or MCL
  (`merged_labelwise_cf`, `recal_holdannot_s2026`), plus `foreign_oof_corpus` on LatOA. Champion proxy
  minus best solo: Effusion +.011, Synovitis −.015.
- `raptor_v4_swa` is #1 by solo macro under every map; its Effusion column goes from #230 of 256 at the
  midpoint to #1 at 0.75; its Synovitis stays ~#240 under every cut.
- Top six by solo macro: 5 of 6 change between R1 and R6 (`stackfit`, `labelwise_dense`,
  `finding_calib_dense`, `calib_dense_hold`, `ramp_dense_gate` enter). Macro ordering Spearman R1 vs R6
  +.96 across 255 members; per column R1 vs picked map Effusion +.43, LatOA +.70, Synovitis +.77, the rest
  ≥ +.87. The champion's five parents rank #32–#82 under every map; the seat structure is what moves, not
  the field.

## What LEDGER75 does NOT fix

- **Synovitis** — its ledger truth is 41 of 57 UNK, below every cut; the recut is flat (+.05). Read it on
  gold-58 (±.08 paired width) or the board. Proxy ceiling: the v2 Effusion rung predicts gold-57
  Synovitis at AUC .702 (v2's own Synovitis rung .694) while the members already read .766 (min .677, max
  .830) — a Synovitis head trained on effusion size cannot reach the members.
- **Contusion** — its coin-flip rung is 0.82 (11 gold+ of 24), above every cut. D26 quoted the 24
  sentences: reports that say contusion / bone bruise run 8 gold+ of 13, reports that say only marrow /
  subchondral edema run 3 of 11 (Fisher p .12, n 24 — hypothesis grade). HIS makes the same split on the
  corpus's 709 rung-0.82 studies: HIS≥.75 in 84% of the contusion-worded (n 296) and 57% of the
  edema-worded (n 329), p 4e−13. So the 0.82 rung is two lexica (`finding_mentions.py:150` lists both
  under one finding), and an edema→0.68 rung split at cut 0.75 would take corpus n+ 1,297 → 968 of 4,406
  and HIS agreement .824 → .898 (.810 → .912 on the 896). Whether the radiologist follows the same split
  is the 24-row question; it is a ruler hypothesis, not a fix. Gold Fracture co-occurs with gold Contusion
  on 10 of 19 positives and is not what separates the rung (p .70).
- **Fracture** — gold n+ 18 at prevalence .310 against .075 on the corpus; hold-10 LEDGER~gold −.75
  under every cut. Both text readers agree (+.76); the 58 does not sample the column. Read it on the
  ledger, never on the 58.

## What this changes

- The ruler is TWO columns, never one repriced: the midpoint column (every ledger number to date) and
  LEDGER75 beside it. Effusion seats are read on LEDGER75; MedMen and Bakers on the midpoint; Synovitis
  on gold-58 or the board; Contusion and Fracture on the ledger, never on the 58.
- Selection caveat: the recut map is a hypothesis fitted on 57 gold rows and 30 members, ten of them
  honest. Its test is the next honest member's Effusion rank landing where its gold rank lands on LEDGER75
  and not on the midpoint column — `hisfluid_hold` is that member.
