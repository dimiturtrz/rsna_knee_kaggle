# The deficit is the member, not the ensemble

*2026-08-26, written at 01:50Z, before `soloseat` scores.*

## What was read

Competition forum topic 735304, "Best single-model score", ten messages, swept with
`python -m kneemri.kaggle.discussion_read --topic 735304`. It costs no submission slot. Self-reported
**single-model board scores**, in the competitors' own words:

| reported | resolution | note the author gave |
| --- | --- | --- |
| 0.942 | 224px | "0.934 @ 224px. Update: 0.942 with some tweaks, still @ 224px" |
| 0.938 | — | "single model, single fold" |
| 0.929 | 224px | "single(5-fold) model, LLM output labels only" |
| 0.926 | 336px | "single(5-fold) model, LLM output labels only" |
| 0.92 | — | "OOF AUC = LB AUC within noise" |
| 0.915 | — | "DinoV2 based" |
| 0.887 | — | the thread's opener, "my current best with DINOv2" |

Two of them volunteer the same opinion unprompted — *"9 days into a competition is a very silly time to
be wasting efforts refining ensemble models"* and *"Public notebooks just ensemble early on to try and
get a high leaderboard score and exposure. Waste of time IMO."*

## What it says about us

`comp/board.jsonl` holds our count curve as **board** numbers, not offline ones:

| members | board |
| --- | --- |
| 1 | 0.914 (`solo_v4`, 2026-08-17) |
| 4 | 0.924 – 0.928 |
| 5 | 0.931 |
| 6 – 7 | 0.936 – 0.939 |

Ensembling buys this project **+0.025** over its own best single. A rival's *one* model reads 0.938 –
0.942, which is our entire six-member champion. Board to board, our member deficit is **0.015 – 0.028**
— at or above the 1.5% house bar, which nothing on the ensemble axis has been for nine days.

That is the whole reading, and it is consistent with three things this repo already measured
independently:

- Nine board entries at 0.936 – 0.939 with count, member identity, fold status, regularisation, encoder
  family and architecture family each moved alone (`2026-08-23_the_ensemble_axis_is_exhausted.md`).
- The offline member-count curve plateaus at eight to ten and spans 0.0007 from five members to
  twenty-two (`record_mix.py`).
- Forum topic 735826, read a day earlier, put the bottleneck at labels first and crop geometry second,
  and explicitly *not* at backbone capacity.

Four lines of evidence, three of them ours, now point the same way.

## The correction I owe this document

My first reading of topic 735304 fixed on one sentence — *"OOF AUC = LB AUC within noise, 0.92"* —
against our offline 0.8598 and board 0.914, and I filed it as a P1: our evaluation table is pessimistic
and has been mis-ordering 130 recorded runs.

It is not news, and the repo says so in two places I should have read first. `board_transfer` already
fits that compression at **+1.82 board points per offline point** across the recorded entries, and
`supervision_gap` already priced the two rulers as close in LEVEL — 0.8391 against the radiologist
annotations, 0.8419 against the reports, on the same 58 studies. A rival's OOF simply sits on a ruler
that reads like the board. The only sound comparison between us and them is board to board, and made
that way the finding is smaller, cleaner and still above the bar. Filed correctly as `knee-nnjb`.

## What does *not* follow

**224px is not a score lever.** Their two best reported 5-fold singles are 0.929 @ 224 and 0.926 @ 336,
and the 0.942 is @ 224 — which invites the conclusion that we are training at the wrong resolution. Our
own five arms on that plateau say otherwise: `native224_long` 0.8490 and `mixup224_long` 0.8505 against
336 members at 0.8525 – 0.8598. On our ruler 224 is about 0.005 *worse*, inside the 0.0062 paired floor.
Flat, not better.

What 224 *is* is a **throughput** lever, and a large one: ~91 s an epoch against ~297 – 361 s for a 336
merged epoch, on an axis where four arms have now read flat against each other. Roughly three times the
experiments per card-hour on a recipe the resolution is insensitive to. Filed as `knee-1xib`, with its
one blocker priced: the merged corpus is decoded at 336 only.

## What is in flight because of this

- **`kneemri-soloseat`** (`knee-lhdb`), pushed 01:44Z, submits when its run completes. `final_ws_e12`
  served alone, to date our own single-member number against 2026-08-17's 0.914. Four pre-registered
  bands live in the kernel docstring; the lowest one is the first board test of whether `--fold -1`
  hurts at serving, which would implicate five of the champion's six seats.
- **`blendsix`** (`knee-4i3m`), ~7 h, gated behind a sixty-study smoke. Six blend-supervised seats
  instead of one. It is a **supervision** change, and "LLM output labels only" appears beside both of
  the thread's highest 5-fold singles.

Neither is priced to close a 0.015 gap on its own. They are the two arms we own that move the axis the
evidence now points at, instead of the one it has finished with.
