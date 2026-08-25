# The 58-study ruler does not obey the competition's own effusion threshold

2026-08-25 · `python -m core.data.labels.severity_boundary` · CPU, no slot, no card

## What was measured

The competition's introduction grades an effusion positive at **moderate or above**. Every label-column
number this project owns is scored against the 58 radiologist-annotated studies — `supervision_gap`
prices each finding against them, `reader_agreement` accepts or rejects a replacement reader by them.
So the question is not whether our reader applies the threshold; it is whether the **annotation** does.

One cross-tab: bucket each annotated report by the size qualifier in its own effusion sentence, read the
radiologist's `Effusion` annotation beside it.

```
28 English annotated studies of 58 annotated (Effusion read from the report)
  small / minimal / trace           12 studies   annotated positive   6
  both, e.g. small to moderate       1 studies   annotated positive   1
  moderate or larger                 3 studies   annotated positive   2
  mentioned, no size word            7 studies   annotated positive   6
  effusion denied                    5 studies   annotated positive   0
buckets holding BOTH classes: small (6 of 12), moderate or larger (2 of 3), unqualified (6 of 7)
```

**The denominator is 28 English studies of 58 annotated**, not 58. 59.9% of the corpus is not English
(`report_language`), and inventing a size vocabulary for seven more languages to answer a yes/no
existence question would put the ruler's own defect inside the tool measuring it.

## What it means

On the side of the threshold the rule calls negative, the annotation is a **coin flip**: 6 positive of
12. On the side it calls positive, 2 of 3. The stated rule predicts its own annotations at chance on the
studies where it is most explicit.

Seven studies are named by the tool. Their sentences read `Small joint effusion`, `Small knee joint
effusion`, `Mild effusion`, `Minimal amount of right knee effusion is present` — all annotated 1 — and
`Moderate joint effusion, distended suprapatellar bursa with thickend synovial tissue are observed`,
annotated 0.

The consequence for us is not a labelling recipe. It is a **ceiling on the ruler**: part of every
measured reader error against these 58 is not error, and no better reader recovers it. Effusion is the
finding where `supervision_gap` shows the largest label-to-model gap (labels 0.8137, model 0.9453,
+0.1317 — `2026-08-14_labels-or-pixels.md`), and this says a slice of that gap is annotation noise at
the boundary rather than headroom.

It also prices the LLM-confidence idea a competitor raised in the same thread: a confidence score over
a report is confidence in **applying a crisp heuristic to a clear word**, not confidence in the finding.
`small` is easy to read and the rule is easy to apply, so the reader is confident — while the process
that wrote `small` rather than `moderate` is the noisy part. Confidence-weighting soft labels on this
axis would weight the wrong quantity.

## Harness suspicion — one of eight reported contradictions was mine

The first run named **eight** studies. Printing their sentences rather than their counts found that
`No significant knee effusion`, annotated 0, had been bucketed *moderate or larger*: `significant` is in
`LARGE_WORDS` and the sentence is a **denial**. That row is agreement, filed as a contradiction by the
tool. `DENIED` now takes denied sentences out of the cross-tab, and the resulting `effusion denied`
bucket is a free control — **5 studies, 0 annotated positive**, which is exactly the agreement a working
ruler should show on the easy cases.

Sensitivity to the softest word in the bucket: dropping `mild` from `SMALL_WORDS` moves 4 studies and
leaves the small bucket at **5 positive of 8**. Still mixed, still ~chance. The finding does not rest on
one vocabulary choice.

## The same question without the regex, on the whole corpus — and what it corrects

`severity_boundary` reads the report, so it is English-only and its Effusion boundary rests on 12
studies. `core.data.labels.level_agreement` asks a neighbouring question with no text at all: for each
finding and each of the reader's five levels, how often did the radiologist agree? That reads all 57
annotated studies present in the reader table, non-English included.

```
  level 0.08  pooled 0.041   per-finding 0.000 to 0.105   (11 findings with 5+ annotated cells)
  level 0.28  pooled 0.136   per-finding 0.000 to 0.341   (9)
  level 0.68  pooled 0.378   per-finding 0.000 to 0.778   (10)
  level 0.82  pooled 0.688   per-finding 0.200 to 1.000   (12)
  level 0.94  pooled 0.958   per-finding 0.600 to 1.000   (7)
```

`MEASURED_AGREEMENT` — the table `calib_span` spent an arm on — holds **one constant per level, pooled
across all twelve findings**. At the mild-YES level that constant is 0.378 and the findings themselves
run from 0.000 (ACL, MCL) to 0.778 (Synovitis).

**Effusion's own rate at that level is 0.481 on 27 annotated cells** — the largest n at 0.68 of any
finding, and independent corroboration of this page's English-only 6 of 12 on a bigger, multilingual
denominator.

**A correction to my own first reading of this.** I wrote that the pooled constant pushes Effusion's
mild positives across 0.5 and inverts them. It does not: 0.481 and 0.378 are on the same side of the
line, so the remap *softens* those cells rather than flipping their ranking. What is true is narrower —
Effusion has both the largest mild-cell mass in the corpus and the largest measured gap from the
constant at meaningful n, which is a magnitude story consistent with `calib_span`'s −0.0212, not a sign
story.

The cells that genuinely invert sit at the **0.82** level, not 0.68: Contusion 0.458 against a constant
of 0.688 on 24 cells is the only one with real evidence behind it (Synovitis, ACL, Medial OA and Lateral
OA also cross, on 5–9 cells apiece).

**What this licenses and what it does not.** The claim that the pooled object is the wrong object is
strongly supported — a single constant standing in for a 0.000–0.778 range is not a calibration. The
claim that we should now *write* twelve per-finding constants is not: most rows rest on 5–9 annotated
cells, and only Effusion at 0.68 and Contusion at 0.82 have enough that their interval clears the
constant with any comfort. Bead `knee-p9oc` carries the arm, and if it runs its pre-registered riser
should be Contusion at 0.82, which is where the ranking actually inverts.

## Provenance

Independently raised by two competitors in discussion topic 735826 (read 2026-08-25 via
`python -m kneemri.kaggle.discussion_read --topic 735826`); one reports the same English template and
the same sentence annotated both ways. Measured here on our own data, and it holds harder than the
forum reported — they claimed at least one moderate annotated 0; this finds the negative side of the
threshold is a coin flip.

Bead: `knee-6qfb`.
