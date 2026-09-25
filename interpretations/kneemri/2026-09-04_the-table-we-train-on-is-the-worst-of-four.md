# The table we train on is the worst of four readers we own

**Date:** 2026-09-04 · **Beads:** knee-wko44, knee-0ib90 · **Card spent:** none · **Slots spent:** none

Every member this project has ever trained reads its supervision from `report_labels_v2.csv`, a lexicon.
The project holds three other reader tables over the same 58 radiologist-annotated studies. Nobody had
scored the four against each other on one ruler. Scored now, with this repo's own
`core.data.labels.reader_agreement`, over 696 cells:

| table | net wrong | false positives | misses | silent |
|---|---:|---:|---:|---:|
| `local_reader_annotated.csv` | **121** | 83 | 38 | 21.0% |
| Qwen3.8-27B NVFP4 (`research/deep_dives/2026-09-03_qwen38_reader_gate_result.md`) | 122 | 48 | 74 | 30.2% |
| `report_labels_v2.csv` — **what we train on** | 140 | 108 | 32 | 25.9% |
| `local_medgemma4b_annotated.csv` | 143 | 103 | 40 | 7.3% |

The supervision under every member is the worst of the four, and its defect has a direction: it
over-calls. 108 false positives against 32 misses. `local_reader` cuts 25 of those false positives at
matched recall — 84.2% against 84.6% — and buys 5.6 points of precision (70.9% against 65.3%) and 3.4
points of raw agreement (82.6% against 79.2%).

This is Wollek et al.'s substitution (arXiv:2306.05997): same trunk, same corpus, change only the
labeler, and the rule-based row reads 0.858 against the learned row's 0.939. No arm in this project has
ever trained on learned-labeler supervision, because no learned reader has ever been run over the 4,406
reports — both learned tables stop at 58 studies.

## The counter-reading, which is load-bearing and which I am not hiding

On the RANKING ruler the ordering reverses. `kneemri.evaluation.supervision_gap` reads `labels - model`
at **+0.0266** for v2 and **+0.0164** for `local_reader`: by that instrument v2 is the better table.

I read that as tie mass rather than reader quality, and the evidence is in the operating-point sweep:
`local_reader` calls 0.9% of cells at the 0.88 cut where v2 calls 10.3%. Its top rung is degenerate, so
its predictions tie far more often, and AUC breaks no ties. This is exactly the effect `table_blend`'s
docstring already documented in the other direction — where quantising a foreign LLM table onto our five
levels dropped it from 0.8927 to 0.8445, *below* ours, and the whole apparent gap turned out to be
resolution. A trainer consumes values and confidences, not ranks, so the operating-point row is taken as
deciding here.

That judgement is the weakest joint in this page. It has been put to Kimi for challenge
(`20260904T2346Z-claude-relabel-timing.md`). If it is wrong, the axis is smaller than it looks.

## Why a rule patch is not the cheaper road, which is what I expected it to be

`overcall_audit` decomposes the lexicon's 108 false positives, and 65% of them are extraction defects
rather than genuine reader-vs-image disagreement:

| class | share of FP | enrichment over true positives |
|---|---:|---|
| subthreshold | 42.6% | +18.9% [+6.6, +32.5] |
| genuine | 35.2% | −17.0% [−27.6, −5.5] (depleted) |
| missed_negation | 11.1% | +9.6% [+4.4, +15.3] |
| unlocated / laterality / hedge / historical | 11.2% | all span zero |

Two classes have intervals excluding zero, so the extraction story is real. But the road from there to a
fix is closed at both ends. The LARGE class has been refused twice by instruments built to test it:
`severity_boundary` found the size qualifier neither separating nor monotone for Effusion, and
`rubric_gap` refused the subthreshold reading for MCL and the whole size family. The CLEAN class is 12
cells — about 1.7% of the annotated cells, and no mechanism that size clears the 1.5% bar on its own.

So patching the extractor has been tried per-class and refused per-class. That is the argument FOR the
substitution rather than against it: a learned reader replaces the whole extractor in one move, and
`local_reader` drops 25 false positives without anyone having written a rule.

## What this does not say

Nothing here is a score. 58 studies is 1.32% of the corpus, every interval on this page is wide, and
`supervision_gap`'s own bands span zero for all three tables. Wollek is chest X-ray, single-label,
German reports; +0.081 is a mechanism floor from someone else's corpus and must never be quoted as ours.

The blocker is arithmetic, not science. `api_labeler.py` times the local reader at ~18 s/report, which
makes a 4,406-report sweep 22 h of card and kills the axis. That figure predates the
`chat_template_kwargs: {enable_thinking: False}` fix that took Qwen3.8 to 1.5 s/report — the same fix
that stops a model spending its budget in a `reasoning` field the caller never reads. A re-timing is the
open question. Under ~3 s/report the sweep is two hours and one matched training arm prices the axis for
real; at a genuine 18 s the sweep goes to Qwen3.8 instead, accepting a different error shape (48 FP and
74 misses against 83 and 38) because a miss asserts nothing while a false positive asserts something
wrong.

Second blocker, and it is embarrassing: **nothing records what produced `local_reader_annotated.csv`**
(knee-0ib90). No doc, no bead, no committed entrypoint — only an mtime of 2026-08-29 16:40. The best
labeler this project owns is currently unreproducible, and a table that cannot be regenerated cannot be
swept.
