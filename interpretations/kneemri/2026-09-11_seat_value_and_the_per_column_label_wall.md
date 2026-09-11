# Seat value tracks representation, and the board gap lives in eight columns

2026-09-11, late. Two findings measured by the researcher (D381, D382) and read here. The first closes
the lever this project has been spending nearly all its card time on. The second is the first lever
anyone has found that is the right SIZE for the gap, and it is the reason the first one closing is not
a disaster.

All numbers are macro AUC at `holdout909`, midpoint-binarised v2 truth, unless a line says otherwise. A
number in this repo is midpoint unless it says otherwise.

---

## 1. Solo value and seat value are different quantities, and seat value tracks representation

Every attribute marginal this project has ever fitted is a SOLO marginal: median-with minus
median-without, over members scored alone. The quantity that actually decides whether an arm is worth
running is its value as an added SEAT to the ensemble we ship, and the two had never been separated.

D381 separated them. Protocol: half-split, 40 readings, LogitAverage, macro AUC midpoint @holdout909,
population rebuilt across both roster shelves to 134 packaged-honest members, candidates drawn from the
0%-exposure sub-pool (n=125). For each attribute, the delta from adding the best member CARRYING it as
a seventh seat to the fixed honest six, against the best member NOT carrying it.

| attribute | solo marginal (0% pool) | SEAT delta | win/40 | times seated in the six |
|---|---|---|---|---|
| merged-rijeka | +0.0117 | +0.0005 +/- 0.0005 | 36/4 | 0 (licence-blocked) |
| labelwise | +0.0138 | +0.0005 +/- 0.0005 | 31/9 | 1 |
| calib-tgt | +0.0109 | +0.0002 +/- 0.0006 | 29/11 | ~4 |
| blend-three | +0.0109 | +0.0001 +/- 0.0005 | 18/22 | several |
| hold-annot | +0.0112 | -0.0000 +/- 0.0007 | 17/23 | ~4 |
| ema | +0.0105 | -0.0003 +/- 0.0005 | 8/32 | several |
| distill | +0.0125 | **-0.0007 +/- 0.0004** | **2/38** | **4** |

**Seat delta does not track solo marginal. It tracks how many times the six already seat the
attribute.** `distill` has the second-largest solo marginal on the honest pool and is the worst
possible seventh seat, losing 38 of 40, because the six carry it four times over (`stack_increments`
x2, `clean_calib`, `seat_clean_distill`) and one more distillate is redundant. The two attributes with
positive seat deltas are exactly the two least-represented ones, and both are worth about half a tick
of noise.

### What this explains, retroactively

Three closed results collapse into one mechanism. The diversity picks failed three times (routing
+0.0008 +/- 0.0033, coverage-greedy -0.0033, addition -0.0004 +/- 0.0007 losing 28/40) and the reading
at the time was that "least correlated with the six" is a proxy for WEAKEST. That was half right.
Least-correlated is a proxy for two things at once -- weakest, and least-represented -- and only the
second one pays. The rule failed because it could not tell them apart. D370's fixed-count reseating
going 0 for 57 is the same fact from the other end.

### What it licenses, and the correction to the earlier reading

D380's closing had argued that saturation licenses spreading attributes one per member. That
over-reached and was withdrawn: k=1 members sit ~10 ticks below k=5 members, and the diversity rule
that would seat a weak spread member has failed three times. But the correction runs the other way too.
"Keep stacking" now fails SPECIFICALLY for distill, which is the one attribute whose seat value against
this six is negative.

The licensed plan is therefore narrow: **keep building stacked members, vary the axes the attribute
table does not index -- teacher identity, label table, fold geometry -- and do not stack more distill
onto anything destined for these six seats.** `chainfour_v5blend` (knee-t9fzu) is the first arm written
under it: `chainfour` with one variable moved, the training label table, to a table seated zero times.

### And the honest size of all of it

The best axis on that table is +0.0005. The board gap is 0.011. Every lever in the table is roughly one
twentieth of the target and sits below the board's own 0.003 estimability floor. This finding is worth
having because it stops the card being spent on seats, not because it points at a seat worth taking.

---

## 2. The board gap is a modelling hole in eight columns and a gold-verified label wall in four

The metric is a macro mean over twelve columns, so a per-column lift divides by twelve and a column
sitting 0.10 below its neighbours carries a hundred times the recoverable mass of a seat swap. D382
asked where the headroom physically is, against the LABEL ceiling rather than against the pool.

Gold provenance, verified: the 469 adjudicated cells are exactly `v2_lone_dissent` intersect
`holdout909`, per column -- every dissent cell in the holdout was adjudicated, so the per-column dissent
base IS the gold n. Overall contradiction 0.8913, tear dissent 277/277, and contradicted cells are
v2=1 -> verdict 0 on 96.9%, i.e. the error process is directional over-call. Per-column "best" is
max-over-304 and selection-optimistic.

| column | gold n | ceiling | best AUC | gap best->ceiling | class |
|---|---|---|---|---|---|
| Medial Meniscus | 111 | 0.849 | 0.9135 | -0.065 | C |
| Lateral Meniscus | 89 | 0.872 | 0.8771 | -0.005 | C |
| Contusion | 120 | 0.861 | 0.8916 | -0.030 | C |
| ACL | 60 | 0.903 | 0.9009 | +0.002 | C |
| MCL | 17 | 0.949 | 0.8274 | **+0.121** | B |
| Lateral OA | 19 | 0.960 | 0.8395 | **+0.121** | B |
| Fracture | 0 | 0.967 | 0.8517 | **+0.115** | B |
| Effusion | 9 | 0.960 | 0.8662 | +0.093 | B |
| PF OA | 11 | 0.960 | 0.8733 | +0.086 | B |
| Synovitis | 6 | 0.961 | 0.8852 | +0.075 | B |
| Medial OA | 21 | 0.956 | 0.9002 | +0.055 | B |
| Baker's | 6 | 0.964 | 0.9317 | +0.032 | B |

Macro: ceiling 0.9301, best-of-304 0.8799, pool median 0.8470.

**Class C -- the label wall, four columns, gold-verified with real n.** Medial Meniscus (v2 contradicted
on 111 of 111 dissent cells), Lateral Meniscus (89/89), ACL (60/60), Contusion (100/120). The pool's
best members score AT or ABOVE the perfect-truth ceiling -- Medial Meniscus's best is **0.065 above it**.
A member cannot beat a perfect-truth ceiling by learning truth; it beats it by learning v2's errors,
which is what the top of the pool has evidently done. Further measured "gain" on these four columns is
more learned noise. Nothing to win here against these labels.

**Class B -- the modelling hole, eight columns.** Ceilings 0.949-0.967 with the best member +0.03 to
+0.12 below.

### The n<30 flag reads backwards here, and class B is stronger than it was written

Every class-B column carries a small gold n (0 to 21), which was flagged as too small to report. That
caution is right in general and wrong in this shape, because the gold n is a **numerator, not a sample
size**. MCL's 17 means "17 of the 909 holdout studies have a contested MCL cell": the rate is 17/909,
denominator 909, precisely estimated, and six-fold smaller than Medial Meniscus's 111/909.

Decomposing what the ceiling is built from,

    epsilon(col) = P(dissent) * P(v2 wrong | dissent) + P(unanimous) * eps_unanimous

the badly-estimated conditional is bounded in [0, 1] by construction, so for MCL the whole first term
is bounded in [0, 0.019] regardless of what those 17 cells say:

    epsilon(MCL) in [0.032, 0.051]   ->   ceiling(MCL) in [0.949, 0.968]

The reported 0.949 is already the worst case of that interval. **`+0.121` is therefore a lower bound on
the hole, and the bound is tight precisely BECAUSE the dissent stratum is small.** A small numerator is
what makes the interval narrow. (Contrast `discord_mask`'s sensitivity at 32 affirmed cells, where the
small n IS the denominator and nothing bounds the quantity -- that one stays unreadable.)

### The single point of failure, and its provenance (D383)

All eight class-B ceilings rest on `eps_unanimous = 0.033` multiplying the ~98% of cells nobody
contested. Provenance, answered: it is **denom60** -- 60 cells drawn uniformly from the five-way
unanimous stratum (v2 plus all four readers agreeing) over the whole corpus, no member signal in the
draw, no overlap with the other packs, **2/60 = 0.033, Wilson 95% [0.009, 0.114]** (D362). That is a
measurement on the population the arithmetic needs, not an assumption. It is also two cells.

The interval spans a regime change, so the table has to be read in two halves:

| eps_u | MCL ceiling | MCL hole |
|---|---|---|
| 0.009 (lower) | 0.972 | +0.145 |
| 0.033 (point) | 0.949 | +0.121 |
| 0.114 (upper) | ~0.869 | ~+0.06 |

**Class membership for MCL, Fracture and Lateral OA survives the entire interval -- they stay class B
even at the top of it. The SIZE ordering against the 0.011 board gap does not.** So the honest statement
is "these three are the biggest holes", and NOT "closing three buys +0.009 board"; the +0.009 may not be
quoted without the interval beside it. Widening this honestly means a bigger unanimous batch, which is a
priced instrument, not a free re-estimate off the same 60.

Separately, Fracture has zero gold cells, so its +0.115 rests on the unanimous leg alone and belongs one
confidence notch below the other seven however attractive its size.

### Two deflations, neither fatal

- "Best AUC" is max-over-304 and carries the project's selection-optimism constant, ~0.011, so every
  gap above is about 0.011 too large. Class membership does not move; sizes do, and Baker's +0.032
  probably vanishes.
- Offline macro does not transfer one-for-one. D372's split-level fit gives slope **+0.29 [+0.09,
  +0.49]** for board >= 0.935 (n=7). So closing MCL entirely is +0.0101 macro and roughly **+0.003
  board** -- one tick. Closing MCL, Fracture and Lateral OA together is +0.030 macro, about +0.009
  board, which IS the gap. One column is not enough. Three is.

### What follows

The four walled columns cap everyone, so if the hidden board labels share v2's error process the leader
pays the same ceiling and that wall is shared, not differential. The differential lives in the eight.
MCL, Fracture and Lateral OA are simultaneously the largest holes and the columns with the most
member-specific variance.

**And the fix is not loss-engineering.** The obvious move is to upweight those three in the objective,
and the project's standing rule forbids it -- keep the objective general, fix needs with balanced data.
The rule is right here for a concrete reason: all three are rare positives, so upweighting buys AUC on
them by trading calibration everywhere else, and macro AUC would hide the trade until the board did not.

---

## 3. Neither obvious explanation survives: the holes are not rarity, and no external corpus fills them

D383 tested the two cheap hypotheses that would each have sent the card somewhere definite, and killed
both. This section is a pair of negatives and it is the most useful thing in this document, because it
is what stops six weeks of acquisition work.

### Rarity is refuted

Positive prevalence per column in `holdout909` against the D382 gap: **Spearman -0.333 (p=0.29, n=12)**
on gap-best, **-0.406 (p=0.19)** on gap-median. Nothing. The individual rows kill it more plainly than
the coefficient does. MCL has **171** holdout positives -- more than Synovitis's 124 -- and is still the
second-worst column in the table. Effusion has **558**, the most prevalent of all twelve, and carries a
+0.093 hole. Medial Meniscus has 482 and sits 0.065 ABOVE its ceiling. Fracture at 68 is the only
genuinely rare class-B column.

So the holes are not data-count holes, and **MCL specifically cannot be fixed by more MCL positives of
the same kind.** It has 596 in train and reads a 0.784 median.

### No external corpus covers class B

| source | usable positives on class B |
|---|---|
| fastmri_findings (1,172 studies, NYU DUA) | MCL **4**, Effusion 142; OA / PF OA / Synovitis / Baker's / Contusion / Fracture **0** |
| mrnet_acl.csv (1,248, Stanford) | ACL 262 only -- no MCL, no OA, no fracture |
| rijeka_acl.csv (736, CC BY-NC-ND) | ACL 189 only |
| labels_hisfluid.csv (4,406) | MCL 767, Lateral OA 1,087, Fracture 314 -- **on our own studies** |

Per column: MCL gains **+4** from the entire external world. Fracture **+0** (fastMRI merges
contusion/fracture into one bone-injury term and splitting it would be guessing). Lateral OA **+0** from
any image corpus. fastMRI+ is an assessed-ABSENCE corpus -- 1,050 to 1,168 negatives per column, with
only high-grade positives recorded, and Synovitis absent from its vocabulary by design. And
`fastmri_findings_effusion.csv` is byte-identical to `fastmri_findings.csv` (md5 match): two filenames,
one table.

`labels_hisfluid` is the apparent exception and is not one. Its 4,406 rows are OUR studies, and its
teacher is the ~80%-exposure foreign_corpus teacher -- a supervision-diversity lever on identical
pixels, which cannot be sold as new data.

**Consequence, acted on: the 231 presigned fastMRI volume URLs are parked, not open.** Pulling them buys
ACL +101 (class C, label-walled), MCL +4, Effusion +142, meniscus +190/+190 (class C), and **zero
class-B volume**, plus an unlabelled in-domain SSL corpus. It adds no new label rows. That thread has
looked like a lever for weeks and is written down here as closed so it stops looking like one.

### What is left is a mechanism question, and the candidate is geometry

With rarity and acquisition both dead, the question is why 596 MCL train positives produce a 0.784. The
candidate on the table (M390, measurement outstanding) is that **the class-B columns are the
PERIPHERAL ones and the recipes crop them off.**

Class C -- where we score at or above ceiling -- is entirely central anatomy: menisci, ACL, and
contusion as a central marrow signal. Class B is not: MCL is superficial and medial, Lateral OA is the
lateral compartment, PF OA is anterior, Baker's is posterior in the popliteal fossa, effusion and
synovitis pool in the capsular recesses. Meanwhile 32 arms train at `--band-low 0.1 --band-high 0.9`,
discarding the outer 10% of the slice stack each side, and 11 train at 0.2/0.8, discarding 20% a side.
If the slice axis runs medial-lateral, the discarded slices are where MCL and the lateral compartment
live, and the members are scoring findings whose pixels were cropped out of their input -- which is
exactly the shape of a hole that more positives do not close.

The contrast needed to test it is **already on disk** (1 arm at 0.0/1.0, 32 at 0.1/0.9, 11 at 0.2/0.8)
and costs zero card. The prediction is falsifiable in the useful direction: class-B AUC should rise with
band width while class-C stays flat, and **a flat class-B row across band widths kills the hypothesis
outright.** The confound to guard is that band width correlates with other recipe attributes across the
pool, so the reading has to come from within-family pairs differing only in band, with a small honest n
preferred over a large confounded one.

## 4. The geometry candidate is dead, three ways, and the band acts through training rather than through inference

Section 3 ends on a hypothesis. D384 killed it the same evening, at zero card cost, from members already
on disk (`research/scripts/band_geometry.py`, per-column AUC @holdout909 midpoint, honest pool n=304 with
56 members of undeterminable band excluded from group medians only). Section 3's closing paragraph should
be read as the record of a question asked, not as a live lever.

**The framing was wrong before the data arrived.** `SLICE_BAND = (0.20, 0.80)` at
`core/data/corpus/slot_reader.py:35` is the inherited default, carried into TrainingOptions, and the
recipe records the RESOLVED band -- so passed-versus-defaulted is unrecoverable from disk. The
no-band-flag group is therefore the NARROWEST crop, not a full-stack control. Section 3 assumed the
opposite and called that group the big control arm. There was never a full-stack control group. The same
docstring records the axis already closed at 12 slices in August: 0.2-0.8 0.8525, 0.1-0.9 0.8565,
0.0-1.0 0.8427, full stack refuted with Baker's, ACL and Contusion named as the losers.

**The prediction inverted on exactly the columns it needed.** Within-family pairs, recipes verified
identical except band (12 slices, 336, v2, e12, max pool):

| contrast | class B | class C |
|---|---|---|
| 0.2-0.8 -> 0.1-0.9 (`lr4e-5_ws` -> `band_span`) | +0.0026 | +0.0065 |
| 0.1-0.9 -> 0.0-1.0 (`band_span` -> `band_full`) | -0.0137 | -0.0140 |
| slices 20 -> 24 at fixed 0.1-0.9 (control) | +0.0016 | +0.0030 |

Widening helps the CENTRAL columns twice as much as the peripheral ones -- the opposite of the
hypothesis -- and going to full stack collapses both classes uniformly with no class separation at all.
The slices control shows the same non-separation, so there is no class-selective geometry term anywhere
in the family. Per column, the two members the story most needed move the wrong way: ACL +0.0148,
Contusion +0.0108, Baker's +0.0105 against **MCL -0.0039 and Lateral OA -0.0038**. And the sharpest
single reading is that **MCL's best band is the NARROWEST** -- 0.7908 at 0.2-0.8, 0.7869 at 0.1-0.9,
0.7823 at full -- with Lateral OA the same shape (0.8178 at 0.2-0.8, best). That is a monotone loss where
a monotone gain was predicted, which is a reversal rather than a null.

**The band is not an inference-geometry knob at all.** A same-weights decode re-banded from 0.2-0.8 to
0.1-0.9 is flat: class means -0.0002, max per-column delta 0.0006, while the probabilities themselves
genuinely move by up to 0.042, so the re-decode really executed. **The band acts through TRAINING, not
through inference.** The outer slices carry nothing the model would use even if shown them, which means
"the evidence was cropped away" was never available as a mechanism regardless of anatomy.

**The orientation worry does not exist.** All 4,407 studies carry Axial, Coronal and Sagittal
(`canonical_metadata.csv`), so the worse variant -- a different anatomy cropped away per STUDY, which
would have looked exactly like unlearnable noise -- is ruled out. What remains is that `band_indices`
samples within ONE series' ordered file list, so a crop trims different anatomy per PLANE (sagittal edges
are the medial and lateral joint line, coronal anterior and posterior, axial superior and inferior),
uniformly across studies. The medial-lateral structure the hypothesis needed is at most one plane of
three. The 2026-08-19 collateral reading (MCL +0.0133 at a wider band) does not rescue it either: that
arm moved slices 12 to 16 together with the band, and its per-column pattern was recorded at the time as
the opposite of what was predicted.

### Where that leaves the eight columns

Three input-coverage explanations have now been tested and all three are refuted: **rarity** (section 3,
Spearman -0.333, p=0.29), **external corpus coverage** (section 3, MCL +4, Fracture +0, Lateral OA +0),
and **geometry** (this section, reversed). The class-B hole is none of the stories in which the model was
never shown the finding. What is left standing is supervision and task difficulty per column.

The live instrument is therefore the one that prices the label side directly: a successor to denom60,
weighted to the class-B columns. It does double duty -- it is the honest widening of `eps_unanimous`,
whose Wilson interval [0.009, 0.114] off **two cells** is the single point of failure for the whole of
section 2, and it tests whether MCL's 0.121 hole is partly the section-2 label wall wearing a smaller
numerator. MCL's unanimous-stratum error has never been audited, and its gold n of 17 is the smallest of
any column carrying gold at all; the hedge-rung family lives in exactly these soft-tissue columns. If
MCL's measured unanimous error comes back materially above 0.033, class B loses its three largest
members and the strategic picture changes. Commissioned in M391; zero card cost.

### A packaging defect found on the way

`merged_blend_cf_n12b0.1-0.9.holdout.npz` is byte-identical to its parent `merged_blend_cf.holdout.npz`
(max|delta| = 0.000000 over all 909x12 cells) -- a cached decode reused under a re-banded name. Its three
resize siblings are genuine (`_r224` 0.638, `_r280` 0.634, `_r392` 0.483), so the defect is isolated. It
was declined and never entered any kernel's `chosen` tuple, so no entry is contaminated, but its decline
reason claimed a "regroup price" that was a no-op by construction and has been rewritten as void. The
generalisable rule: **a variant npz reading exactly 0.000000 against its parent is a packaging bug, never
a null result** -- this section's own re-decode number, up to 0.042, is what proves it.

---

## Population note

D382 ran on the honest pool at n=304 @holdout909; D381 on 134 packaged-honest members with candidates
from the 0%-exposure sub-pool n=125, both rosters walked. `greedy_halfsplit` and `diversity_seventh`
ran on 117, a single-shelf walk, and still may not be quoted until rebuilt.

One process note worth keeping. D381 reported `distill_stack_clean` as truncated at epoch 3 off a
`metrics.jsonl` that was still being written; the arm was live at epoch 4 of 12 with its log mid-epoch
at `step 550/582`. A live run and a dead one are indistinguishable on disk and `convergence_check`
cannot separate them -- only the process table can, and MSYS `ps` shows no argv, so the check is
`Get-CimInstance Win32_Process -Filter "Name='python.exe'"` filtered on `CommandLine`. The reading that
mattered, `S - S'` for the exposure pair, was not broken, only not yet takeable -- and it was taken
hours later: `distill_stack_clean_s2026` converged at **0.8645**, +0.0011 ABOVE its ~80%-exposed twin
`distill_stack_max_s2026` at 0.8634. Removing teacher exposure cost the member nothing, which is the
first member-level price for the fourth honesty filter and agrees with the ensemble-level substitution
estimate of ~0.0004 through two of six seats. Exposure and the fold-0 LEAK (+0.0051) are different
quantities and only the second was ever inflating anything.
## 5. The answer section 4 was waiting for: MCL's hole is modelling, not wall

Section 4 closed on a commissioned instrument and an explicit branch -- "if MCL's measured unanimous error
comes back materially above 0.033, class B loses its three largest members and the strategic picture
changes." It did not. It came back at **zero**.

`classb120` was built, sealed and pre-registered by the researcher within seven minutes of the commission
(D385): three columns x 40 rows, drawn uniformly inside each column's five-way unanimous stratum
(`v2 + flight + lixin + steven_blend + steven_full`, binarised at 0.5), over the same five-table UID
intersection of 4,406 studies that denom60 used, no member signal, zero overlap asserted against all six
prior packs, seed 20260911. Pack sha `cc6fddb6c09e213e`, key sha `aa177af27706d982`. All 120 cells were
adjudicated blind from the free-text reports -- nine languages -- and `adjudication_key.csv` was opened
only after the last of the 120 answers was on disk.

| population | k / n | rate | Wilson 95% |
|---|---|---|---|
| MCL | **0 / 40** | 0.0000 | [0, 0.0876] |
| Lateral OA | **0 / 40** | 0.0000 | [0, 0.0876] |
| Fracture | 1 / 40 | 0.0250 | [0.0044, 0.1288] |
| classb120 pooled | 1 / 120 | 0.0083 | [0.0015, 0.0457] |
| denom60 (prior) | 2 / 60 | 0.0333 | [0.0092, 0.1136] |
| **pooled, n = 180** | **3 / 180** | **0.0167** | **[0.0057, 0.0478]** |

The pre-registered null was ~6/180 with [0.015, 0.071]. The observed 3/180 sits just under that interval's
floor: the unanimous stratum is CLEANER than denom60's two cells implied, not dirtier.

**The single disagreement is an adjudicator boundary call, not a demonstrated label error.** `c0113` reads
"subacute osteochondral injury at the medial patellar facet (about 1.3 x 1.5 cm)" plus a bone bruise at the
lateral femoral condyle, with no free fragment and the word fracture nowhere in the report. It was scored 1
on the reasoning that an osteochondral injury breaches subchondral bone; the tables scored 0. That is a
definition of the Fracture column, not a defect in it -- so the honest reading of this batch is 0-to-1
errors in 120, and the pessimistic end is the one tabled above.

**What it decides.** MCL's unanimous error is 0/40 with a Wilson upper of 0.0876 -- not materially above
0.033, and nowhere near the class-C regime (Medial Meniscus 111/111, Lateral Meniscus 89/89, ACL 60/60).
**MCL's +0.121 hole is real modelling headroom, and so is Lateral OA's.** Class B is attackable. Section 4
listed three refuted input-coverage stories -- rarity, external corpus coverage, geometry -- and named
supervision and task difficulty as the survivors; this section removes the label wall from the supervision
half AT UNANIMITY, which is the half that would have capped the columns outright.

**And section 2's single point of failure is closed on the high side.** That section's whole per-column
ceiling table rested on `eps_unanimous` = 2 cells, Wilson [0.009, 0.114], and the class-B ordering flipped
depending on which end of that interval you stood on. At n=180 the interval is [0.0057, 0.0478], about 2.4x
narrower: **0.114 is now excluded**, so any reading that was only true at the high end is dead, and the
0.033 reading survives as the conservative one. The D383 rule stands -- `+0.009 board` is never quoted
without its interval beside it -- but the interval it now carries is a different object.

**What is left.** Unanimity is 69.34% of the corpus. The disagreement stratum (`W_D` = 0.3066) has never
been audited on any class-B column, and it is exactly where a GRADED soft-tissue finding would hide: MCL is
a grade I/II/III call and the reports are written in grades, so tables binarising at 0.5 will disagree
precisely on the rungs. A disagreement-stratum successor on the same three columns is commissioned in M392,
scored against the majority sign and, if free, against each table separately -- a table carrying the
disagreements names itself as the one to drop. Adjudication cost is now known and small: one agent, ~25
minutes, zero card.

---

