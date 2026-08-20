# The reader is right 93% of the time, and the label it never gets wrong is the one we score worst on — 2026-08-20

## What was run

An Opus subagent read the first 100 reports of the corpus against their current 12 attributions — 1,200
cells — carrying the incumbent labeler's **verbatim rulebook**, lifted from `api_labeler.py` (found the
same day inside pilkwang's CC0 dataset; see `2026-08-20_what_is_left_after_decode_and_data.md`). Using
the incumbent's own rules is what makes a disagreement an ERROR rather than definitional drift, and it is
the only reason the count below means anything. The agent was told explicitly not to manufacture
disagreements and was given an UNCERTAIN escape hatch, because a contest agent that wants to look useful
will find 30% of anything.

**Result: 1,115 agreed (92.9%), 48 contested (4.0%), 37 uncertain (3.1%).**

Raw per-study tables: `2026-08-20_label_contest_pilot_tables.md`, beside this note.

## The 4% is not the finding. The zero is.

| label | our holdout AUC | contested / 100 |
|---|---|---|
| **MCL** | **0.788–0.793** (worst) | **0** |
| ACL | 0.874–0.877 | 0 |
| Fracture | 0.796–0.809 (2nd worst) | **19** |
| Contusion | 0.863–0.875 | 9 |
| Lateral OA | 0.810–0.820 | 6 |
| everything else | — | 0–3 |

**MCL is the worst-scoring label in the corpus and its labels are clean.** ACL, well-scored, also clean.
`experiments/arms/resolution.sh` argued from occlusion evidence that MCL is hard to SEE — a thin ligament
at the medial margin of the field, read coronally, at 0.39 mm per pixel — and rarity and label noise were
both tested and refuted as explanations there. This pilot is a fourth, independent line arriving at the
same place from the opposite direction: MCL is not hard to READ. Whatever bounds it is not supervision.

Fracture is the reverse and is therefore the WEAKER case, not the stronger one: it has bad labels AND a
bad score, so it can never be the clean experiment. The clean experiment is MCL, and it says no.

## What the errors actually are, and why the direction matters more than the count

| mode | cells |
|---|---|
| NO → UNK (a denial the report never made) | 23 |
| UNK → NO | 13 |
| UNK → YES | 3 |
| NO → YES | 3 |
| YES → UNK | 2 |
| YES → NO | 1 |
| severity only | 3 |

**The bias is one-way: the reader fabricates DENIALS, never positives.** NO→UNK is the single largest
bucket and it is not a sign error — it is a WEIGHT error. Those cells train at value 0.08 with weight
**0.85** where the rulebook puts them at 0.28 with weight **0.05**. A cell that should be nearly switched
off is being trained at near-full confidence.

Two questions produce 36 of the 48:

1. **No stable rule for what a normal-bone sentence denies.** The same template gets different answers:
   Turkish signal-only `sinyal intensiteleri doğaldır` → Fracture NO in studies 008/037 but UNK in
   032/046; Bulgarian structural `нормална форма` → NO in 054/085 but UNK in 007/036.
2. **A finding the report never names, answered NO** (23 cells), almost all Fracture / Contusion /
   Baker's — the findings a normal knee report simply does not mention.

Both are rulebook ambiguities rather than reading failures, which is why they cluster on templates rather
than scattering.

Also worth recording, because it bounds any FUTURE re-read rather than this one: **bilateral or
compartment-unspecified spans answered for one side only** (6 cells — e.g. 037 `Medial ve lateral
menisküste grade II dejenerasyon` scored medial and left lateral UNK, with the lateral named in the very
sentence used for the medial), and an **osteochondral lesion → compartment OA** convention that is simply
undecided (UNK in 018/022/052, YES in 015/044/054/061; 9 cells parked as uncertain).

## The multilingual half, measured rather than assumed

| language | contested / cells |
|---|---|
| Bulgarian | 6 / 48 (12.5%) |
| Turkish | 12 / 156 (7.7%) |
| Croatian | 6 / 96 (6.3%) |
| Spanish | 7 / 252 (2.8%) |
| English | 13 / 480 (2.7%) |
| German | 0 / 24 |

Non-English 4.9% against English 2.7%. The excess is real but it sits in the three languages whose
normal-boilerplate bundles bone and cartilage into one clause — i.e. it is failure mode 1 again, in the
languages where that clause is most compressed, rather than a general reading deficit. This is consistent
with the 2026-08-19 measurement that the LLM reader is NOT blind to the non-English corpus (the regex
path is) and narrows it: the reader's residual multilingual weakness is one specific sentence shape.

## Against the one ruler that does not move

The 58 image-read annotations (57 after the merge) give a false-denial rate per label — a report cell at
0.08 where the image read says the finding is present:

| label | NOs | false denials |
|---|---|---|
| Fracture | 12 | **0 (0.0%)** |
| ACL / MCL / Effusion / Synovitis / Baker's | 21 / 33 / 5 / 0 / 12 | 0 |
| Lateral OA | 19 | 2 (10.5%) |
| Lateral Meniscus | 20 | 2 (10.0%) |
| Contusion | 11 | 1 (9.1%) |
| PF OA | 19 | 1 (5.3%) |
| Medial OA | 21 | 1 (4.8%) |
| Medial Meniscus | 23 | 1 (4.3%) |

**These do not contradict the pilot and must not be read as a refutation of it**, for two reasons, and
the second is the one that matters:

- **The denominator is too small to resolve the question.** Fracture's 0/12 carries an interval of
  roughly [0%, 26%], which contains the pilot's 19% comfortably. Nothing is being refuted at n=12.
- **This ruler is blind to the pilot's dominant error by construction.** NO→UNK does not change which
  side of 0.5 the cell sits on, so a false-denial count cannot see it unless the finding was genuinely
  present. The two measurements are on different axes: the pilot measures conformance to the RULEBOOK,
  this measures conformance to the IMAGE.

## The arithmetic that decides it

The competition metric is an **unweighted mean of twelve AUCs**. 58% of the contested cells sit on the
bone axis (Fracture + Contusion), which is 17% of cells. Price a generous fix — Fracture 0.80 → 0.85,
Contusion +0.02, everything else unchanged:

    (0.05 + 0.02) / 12 = +0.006 on the mean

That lands **at the 0.0062 holdout noise floor** and is a twentieth of the project's 1.5% bar. And it is
generous twice over: it assumes correcting 19% of one label's cells buys five points of AUC on it, and it
assumes the correction helps at all — which is the assumption the ledger has already tested and lost.
`--calibrated-targets` remaps the level values to their measured agreement rates and scored **−0.0020**
as `calib_span` on the recipe we ship, with every pre-registered clause inverting. Correcting label values
on this corpus has been tried once and cost score.

## Verdict

**A full LLM re-read of all 4,407 reports is not the lever, and this pilot is what says so rather than a
prior.** ~1.4M input tokens to move a number whose ceiling, generously priced, is the noise floor.

What the pilot DOES settle, and it is worth the 62k tokens it cost:

- **Supervision is not what bounds MCL**, our worst label — 0 errors in 100 studies. Combined with the
  occlusion evidence in `resolution.sh` and the rarity / label-noise refutations there, the remaining
  explanation for MCL is that we cannot resolve it. That is an IMAGING axis and it already has a named arm.
- **The label defect that exists is a WEIGHT defect, not a value defect** — one-way fabricated denials on
  findings a normal report does not mention. Any future re-read should fix the RULEBOOK's two ambiguities
  first (what a normal-bone sentence denies; unmentioned = UNK, never NO), because 36 of 48 errors are
  those two questions and neither needs a better model to fix.
- **The reader's multilingual residue is one sentence shape**, not a language deficit.

## The one thing this DOES confirm: knee-dtk's split is real

knee-dtk partitioned the twelve findings on 2026-08-14 by whether the REPORT or the MODEL is the weak
part, using the 58 image-read annotations: label-bottlenecked = Effusion, Contusion, Synovitis, Medial
OA, Fracture, Baker's (the model already beats its own labels there); model-bottlenecked = ACL, MCL,
Medial Meniscus, Lateral Meniscus, Lateral OA, PF OA.

The pilot never saw that partition. Its errors fall across it **3:1**:

| half | contested | rate over its own 600 cells |
|---|---|---|
| label-bottlenecked | 36 / 48 (75%) | **6.0%** |
| model-bottlenecked | 12 / 48 (25%) | **2.0%** |

Two independent measurements — 58 studies read against IMAGES, 100 studies read against a RULEBOOK —
agree on which six findings carry weak supervision. That is worth more than either alone, because
neither could produce this by construction: knee-dtk's split cannot see rulebook conformance, and the
pilot was never shown an AUC.

It also sharpens the verdict rather than softening it. The label errors sit **exactly where a better
reader would pay** — and the whole axis still prices at +0.006. knee-dtk's rule ("a better label reader
pays HERE and nowhere else") is confirmed and simultaneously shown to be worth very little, because the
label half's findings are the ones we already score well on.

And the converse holds at the sharpest point: MCL and ACL are the two labels with ZERO contested cells,
and both are in the model half. The partition and the pilot name the same two findings as pure vision
problems.

## The objection that "0 contested" might just mean "never mentioned" — measured

Agreement on ABSENCE is not evidence of correctness, and a label the reports simply never discuss would
produce a clean contest score for the worst possible reason. That is failure mode 2 of this very pilot,
so it deserved a measurement rather than an argument. **Denominator: all 4,406 corpus rows, and every
one of the twelve columns is populated on every row**, so these are rates over the corpus and not over a
subset that happened to parse.

| label | UNK % | NO % | YES % |
|---|---|---|---|
| **Synovitis** | **84.2** | 2.6 | 13.2 |
| **Fracture** | **56.4** | 36.5 | 7.1 |
| Lateral OA | 33.1 | 42.3 | 24.7 |
| Medial OA | 25.5 | 37.7 | 36.7 |
| Contusion | 20.8 | 49.8 | 29.4 |
| PF OA | 18.5 | 36.3 | 45.2 |
| Lateral Meniscus | 10.0 | 64.5 | 25.4 |
| Effusion | 9.9 | 29.6 | 60.5 |
| **MCL** | **9.8** | 72.8 | 17.4 |
| **ACL** | **8.1** | 64.1 | 27.8 |
| Medial Meniscus | 5.7 | 41.1 | 53.3 |

(Baker's Cyst is absent from this table on a column-name mismatch, not a missing measurement.)

**For MCL and ACL the objection fails, and it fails at the sharpest point.** They are the two labels with
ZERO contested cells and they carry the LOWEST UNK rates of all eleven — 9.8% and 8.1%. Their reports
overwhelmingly say something definite. MCL's 72.8% NO rate is the highest in the corpus: the reports
address the medial collateral ligament and call it intact. The reader agreeing with that is agreeing
with a real statement, not with a silence.

**But the objection RELOCATES and lands harder somewhere the write-up above never named.** 84.2% of
Synovitis and 56.4% of Fracture cells are UNK — 3,710 and 2,485 studies with no information on that
finding at all. Over half the corpus carries nothing about fracture. That is a supervision hole far
larger than the 4% contest rate this pilot was built to measure, and it is invisible to a contest, which
can only score cells where the reader made a call.

**And then it declines to explain our scores, which is the part worth keeping.** If missingness capped
performance, the ordering would show it. It inverts:

- **Synovitis has the worst supervision coverage in the corpus and scores 0.873** — mid-pack, above
  seven other labels.
- **MCL has the third-best coverage and scores 0.788–0.793** — the worst of the twelve.

So the hole is real, it is much bigger than the contest rate, and it is **not** what bounds us. It also
has a mechanism for why it costs less than it looks: under `--weight-floor` an UNK cell is not dropped —
it trains at value 0.28, weight 0.2875, a soft negative. Synovitis's 13.2% positives are learnable
against a quiet negative background, which is what a 0.873 on 84% missing data looks like.

This does not make missingness harmless. It makes it a **ceiling question rather than a floor question**,
and it is the one supervision axis this pilot did not price. Filed as its own bead rather than folded in
here, because pricing it needs a different measurement than a re-read.

## What this does NOT say

It does not say the labels are correct — 93% conformance to a rulebook is not 93% correctness, and
`Supervision.MEASURED_AGREEMENT` already records that a 0.68 cell is positive only 37.8% of the time
against the image. It says re-deriving them with the same class of reader is not worth the pass.

It does not close knee-mxj by refutation; it closes the branch of it that a re-read would serve. The bead
had already lost its other branches — to an audit (the grade/certainty split it was designed around is
shipped, and `__verdict` is 100% redundant with `value` under the bijection) and to the ledger
(`calib_span`). What survives is small.

It also does not generalise past 100 studies with confidence. 4.0% carries a binomial interval of about
±1.1% on 1,200 cells before any clustering correction, and cells are clustered by study, so the true
interval is wider. The per-label counts — Fracture 19/100 — are the noisiest numbers here, and the
argument above deliberately does not rest on their precision, only on the contrast between 19 and 0.
