# The rubric our reader was never given

2026-08-27 · `python -m core.data.labels.rubric_gap` · no GPU, no slot · beads `knee-rwcy`, `knee-ufz5`

## The host said what the labels mean, and it is not what our reader was asked

Discussion topic 733491, answered by the competition host today, and topic 733343 (the Overview thread
it points to), which publishes the image-review criteria for all twelve findings:

> For meniscal tears, the target is a definite tear; intrasubstance degenerative signals that do not
> reach the articular surface are negative. … marrow edema, cartilage findings, and terminology in the
> narrative or impression do not by themselves determine the contusion, osteoarthritis, or other
> challenge labels. Those labels reflect the standardized image-review criteria. … A report may
> therefore use language that appears inconsistent with the corresponding image-based label because the
> original report and the image-based labels were generated through different processes.

Every target this project trains on is derived from the report. `ReadingPrompt.system()` instructs:
*"For each of the twelve findings, say what THE REPORT SAID about it. Do not diagnose, do not infer from
other findings."* That is the right instruction for **transcription** and the wrong one for **this
target**, because the target is not the report. It is a two-reader image review with a third
adjudicating, under criteria the host has now stated. The words `intrasubstance`, `degenerativ` and
`definite` appear nowhere in `core/` or `kneemri/`.

This is a **faulty-DATA** finding in the house taxonomy — the input the method needs is mislabelled —
and not a faulty test or a faulty idea.

## How big

Three of the twelve published rules have the same shape — a MENTION of the structure, a vocabulary the
rubric ACCEPTS, a vocabulary it names and calls NEGATIVE anyway, and a denial that must be subtracted
before either is searched for. `core.data.labels.rubric_gap` holds one `RubricRule` per finding and reads
our derived target beside the bucket. The stated rules supply the correct answer for the subthreshold
class without needing any annotation, so the denominator is the corpus rather than the 58.

The other nine turn on a size the report states in its own units ("moderate or large", "one centimetre
of >50%-thickness loss"). That is `severity_boundary`'s shape, and it needs the annotations.

```
                                 subthreshold cells   ours positive   contradicted   of that finding's cells
Medial/Lateral Meniscus                        196             165            165                     1.87%
ACL                                            102              98             98                     2.22%
MCL                                            213             185            185                     4.20%
                                                                              ---
                                                                              448
```

**448 cells, and every one of them a FLOOR.** The tool reads 40.1% of the corpus; the other 59.9% is
Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch or French and carries the same wording in its
own language. If the class rate is language-flat the corpus figures are nearer 4.7% / 5.5% / 10.5%. The
rates are stated over the corpus, not over the subset that parsed, because a rate computed inside the
subset that happened to parse is not a rate over the corpus.

Concentration matters more than the raw share. **165 of 196, 98 of 102 and 185 of 213 — 84%, 96% and 87%
— are flipped**, and all in one direction, onto the positive class. That is not symmetric noise a bigger
model averages out; it is a bias that moves the decision boundary. Four of the twelve columns carry it.

## The annotations agree with the host on two arms and hedge on the third

Of the 58 annotated studies, 27 are English. Bucketed the same way:

```
                       subthreshold cells   annotated positive   ours positive     accepted bucket
Meniscus                                3                    0               3     23 of 27 agree
ACL                                     1                    0               0     10 of 12 agree
MCL                                     7                    3               7      2 of  2 agree
```

Meniscus is the clean one: every subthreshold cell is annotated 0 while our target calls all three
positive, and in the other direction the accepted bucket agrees 23 of 27 — so the ruler is not simply
strict, and the disagreement is specific to the class the host named. The pilot table
(`2026-08-20_label_contest_pilot_tables.md`, row 33) had already flagged one of these by hand as a
READER error; it was a RUBRIC error all along, and studies 059, 062, 088 and 090 are the same shape.

**MCL is the largest arm and the least trustworthy one, and the annotation says so.** 3 of its 7
subthreshold cells are annotated POSITIVE — the ruler and the radiologist disagree 43% of the time on
exactly the class the count rests on.

The flag paid immediately, and it paid in the ruler rather than in the rule. Reading topic 733343
verbatim rather than from a summary of it gives the MCL criterion as *"a high-grade partial or complete
acute tear … **with disrupted fibers and edema within and adjacent to the ligament**"* — so edema belongs
to the POSITIVE class, and the first version of this ruler had `oedema|edema|thicken` in MCL's
SUBTHRESHOLD vocabulary. It was reading the accepted class as the negative one. Corrected (edema and
thickening dropped from subthreshold, `disrupt` added to accepted), MCL falls **207 → 185** and the total
**470 → 448**.

The 3-of-7 annotation disagreement is UNCHANGED by that fix — those seven cells are named-but-under-
threshold on other words. So MCL stays flagged: treat 185 as an upper bound whose true share is
unmeasured, and do not quote the 448 total without this line attached. ACL's 1 annotated cell settles
nothing either way.

The general lesson is the one the house rule already states and I broke anyway: I wrote a vocabulary for
a published rule from my own knowledge of MCL injury instead of from the published rule. The criteria now
live once, quoted, in `core.findings.CRITERIA`, and both the ruler and the reader's prompt render from
there — so a paraphrase cannot drift from the thing it is scored against.

The honest summary: **meniscus is measured, ACL is plausible and thin, MCL is flagged.**

## And the prompt was broken in a second way nobody had read

Writing the rubric into `ReadingPrompt.system()` meant printing the rendered prompt for the first time in
this project's life. It rendered the grade vocabulary like this:

```
Grades: absent (WHAT THE REPORT SAID about one finding — never how sure the model feels about it. …),
unclear (WHAT THE REPORT SAID about one finding — …), possible (WHAT THE REPORT SAID …), …
```

`system()` read `grade.__doc__` for each member of `ReportGrade`, and **a bare string under an enum
member does not become that member's `__doc__`** — it falls back to the CLASS docstring. So every reader
run this project has ever made was handed the class docstring five times over, about a thousand tokens of
repeated prose, exactly where the definitions of `absent` / `unclear` / `possible` / `present` / `severe`
were supposed to be. The model was told the five WORDS (they are in the JSON instruction) and nothing at
all about what they mean.

It is silent by construction: the reader answers in the right vocabulary either way, so no parse fails,
no schema breaks, and no metric names it. It surfaced only because someone read the prompt.

The definitions now live in a `WORDING` dict beside the enum, the same shape as `LEVEL` two lines below
it, and `test_system` asserts both that each definition appears and that the class docstring does not.
The corrected prompt is **775 tokens against the broken one's 1037** — twelve published criteria added,
and it still came out a quarter shorter.

**This re-prices `knee-ufz5` upward, and it is a genuine unknown.** The re-read fixes two defects at
once: the reader gains the host's thresholds AND, for the first time, a definition of its own grades. The
`possible` rung in particular — the one carrying 6461 cells at 0.68 — was being assigned by a model that
had never been told what `possible` means. How much of the ladder's measured behaviour is a fact about
hedged reports and how much is a fact about an undefined word is not something this project can currently
answer.

That does not reopen the ladder's CALIBRATION (see the last section — 0.68's worth is measured against
image annotations and is independent of how a cell earns the rung). It does mean the reader's measured
ceiling — `blendlabels`' +0.0034, and every per-label supervision gap read off it — was measured with an
instrument that was never given its own units.

## Why the blendlabels precedent does not price this down

`blendlabels` measured that a better reader buys **+0.0034** on the mean and **+0.0070** on the six
label-bottlenecked findings, and the ledger reads that as *"the label lane is confirmed LIVE and
confirmed SMALL"*. That arm blended three readers of the **same prompt** — it reduced reader VARIANCE.
Averaging three readers who all read degeneration as a tear fixes none of these 163 cells. The
mechanisms are different, and the ledger's own per-label numbers say so: under blending, Lateral
Meniscus went **−0.0105** and Medial Meniscus **−0.0083**, the two columns this defect sits on. Variance
reduction made the meniscus columns slightly worse, which is what you expect when the residual is shared
bias rather than scatter.

Lateral Meniscus also carries this project's worst supervision gap (−0.193).

## What it is worth, and what it is not

Do not oversell it. Four of twelve columns, so a per-column gain divides by three on the mean; and
`calibsix`'s measured board pass-through is 23%. An AUC gain of 0.03–0.05 on each of the four would be
about +0.010–0.017 on the twelve-label mean and +0.002–0.004 on the board — **directional under the 1.5%
bar on the board, and only at the bar on the holdout mean if all four move and MCL survives its own
caveat.** It is worth doing anyway for three reasons the bar does not capture: it
is a data fix rather than a knob, it is the first correction this project has that a HOST statement
licenses rather than a measurement of our own, and it composes with everything already banked.

## Two fixes, and only one of them is right

**(a) Post-hoc, English only.** Write `report_labels_v3` with the 470 cells set to absent. Minutes, no
card. Rejected as the end state: it corrects 40% of the corpus and leaves 60% wrong, which converts a
uniform label error into a **language-dependent** one. That is a new defect, not obviously smaller than
the one it fixes.

**(b) Re-read with a rubric-bearing prompt** (`knee-ufz5`). The host's twelve criteria plus the global
*"ambiguous or borderline findings were graded as negative to favor specificity"*, added to
`ReadingPrompt.system()`, then re-read all 4406. It reaches every language because the READER does, not
a regex, and it covers the nine findings this ruler cannot see as well as the three it can. Costs one
LLM pass over the corpus and no submission slot.

(b) is the one to run, gated on the card clearing its current queue. The measurement above is what makes
it priceable: the re-read is scored by whether those 470 cells move — meniscus's 165 in particular,
which is the arm the annotations back — and a re-read that leaves them where they are has failed loudly
instead of silently. Second gate, which must not regress: `kneemri.evaluation.annotated_readings`.

## What this does NOT reopen

The confidence ladder. `0.68` for a hedge sits above the midpoint on 6461 cells (12.2% of the label
matrix), which looks like it contradicts *"on the fence → negative"* — and it does not. `level_agreement`
already measures that rung's pooled constant at **0.378** on our own annotated cells, consistent with
pooling (χ² p=0.100) and settled on disjoint halves (0.9σ); and `calib_span` already spent an arm
remapping every rung to its measured agreement, returning **−0.0020** with its per-label prediction
inverted. This finding changes which RUNG a report earns. It says nothing about what a rung is worth.
