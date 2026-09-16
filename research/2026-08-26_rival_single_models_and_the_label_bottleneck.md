# Rival single models, and what the field says the bottleneck is

**Read 2026-08-26 22:11Z** via `python -m kneemri.kaggle.discussion_read --live`, topics 735304 and 735826.
Costs no submission slot. Theirs, not ours — every number below is self-reported by a competitor and
unverifiable; treat them as claims with a plausible-lie floor, not as measurements.

## The number that matters

Self-reported SINGLE-model public-board scores, topic 735304:

| claim | resolution | note |
|---|---|---|
| 0.942 | 224 px | "0.934 @ 224px. Update: 0.942 with some tweaks, still @ 224px" |
| 0.938 | — | "single model, single fold" |
| 0.934 | 224 px | same poster as the 0.942, before tweaks |
| 0.929 | 224 px | 5-fold, "LLM output labels only" |
| 0.926 | 336 px | 5-fold, "LLM output labels only" |
| 0.92  | — | "OOF AUC = LB AUC within noise" |
| 0.915 | — | DINOv2-based |

**Our best six-member ensemble scores 0.939.** A rival's single model is claimed at 0.942, and two more
sit at or within noise of our ensemble. Our best single-model board entry (`soloseat`) reads 0.925.

> **UPDATE 2026-09-01 — the numbers above are correct as read on 2026-08-26, and the ordering they
> support has since flipped.** Recomputed over the completed `comp/board.jsonl` (87 rows, `scope:
> excluded` dropped, joined on the `members` field): our best six now reads **0.943** (`champviews_v1`,
> 2026-08-31) and our best seven **0.944** (`stackseat_v1`, the current champion's family). So a rival's
> claimed 0.942 single no longer beats our ensemble — it sits inside it. What does NOT change is the
> localisation this section draws: their SINGLE is still level with our SIX, and our own best single is
> still 0.925. The member gap is unclosed; only the sentence "their single beats our ensemble" expired.
> The 0.914 quoted in the next paragraph was also right on 2026-08-25 — `soloseat_v1` scored 0.925 the
> following day. See knee-macc for the count-curve recompute this came out of.

**This localises our gap in MEMBER QUALITY and not in ensembling, for the second time.** Every ensemble
number we own is measured inside our own pool, so it cannot see this; only the forum can. It is also the
second independent statement of it — topic 735304 said the same on 2026-08-25 with a 0.915-0.942 spread
against our then-best single of 0.914.

Two of them volunteer the corollary directly: *"9 days into a competition is a very silly time to be
wasting efforts refining ensemble models"* and *"Highly doubt the top results are ensembles — look at the
efficiency LB, many high-scoring submissions have decent run times."* The efficiency board is checkable
evidence for that claim rather than an assertion, and we have not looked at it.

## What the field says the bottleneck is (topic 735826)

The question was posed as A labels / B preprocessing geometry / C backbone capacity / D validation /
E other. Nobody argued C. Two independent claims worth keeping:

- **Larger encoder buys nothing.** One competitor froze a `(4407, 6, 9, 224, 224)` uint8 cache and swapped
  ResNet34 → DINOv2-small with slot attention, holding data and labels fixed: OOF BCE went 0.664 → 0.674,
  i.e. WORSE. Matches our own `knee-` scaling result (+0.0011) and our v3base/large arms. Backbone size is
  a closed axis, theirs and ours.
- **Report labels contradict image labels near the severity boundary.** Filter reports whose effusion
  sentence carries a small-size qualifier ("small", "minimal", "trace") and the group holds both
  `Effusion = 0` and `Effusion = 1`; one poster found **two reports with the same English template and the
  same sentence carrying opposite labels**, and at least one "moderate effusion" report labelled 0. So it
  is not a threshold at all — it is reader disagreement written into the gold set.

The sharpest consequence stated, and it is one we can act on: **asking an LLM for its confidence is
measuring its confidence in APPLYING A HEURISTIC TO TEXT, not the uncertainty in the diagnosis.** It will
be very sure "small effusion" means 0, while the process that chose the word "small" over "moderate" was
itself noisy. Our calibrated-target arms inherit exactly this defect.

And the diagnostic worth running: **score your label extractor against the 58 gold studies, then score
your image model against the same 58.** If the image model does not beat its own teacher there is
modelling headroom; if it beats it by a lot, the labels are the ceiling. We have
`kneemri/evaluation/annotated_readings.py` and have never run this comparison in that direction.

## What this changes here

Nothing about tonight's authorised slot — `kneemri-resseat` is already argued and already an explicit bet.
It changes what the next block of card time is worth spending on:

1. The label lane (`knee-pju4`, P0) is not one lever among several. It is the lever the field agrees on,
   and 27.3% of our supervision cells (14,445 of 52,872) are a constant rather than a read.
2. The severity boundary is a real, named, reproducible defect with a filter that finds it — see
   `interpretations/kneemri/2026-08-25_effusion-threshold-ruler.md` and `core/data/labels/severity_boundary.py`.
3. The efficiency leaderboard is unexamined evidence about whether the leaders ensemble at all.

Not acted on here, and none of it is verified against a primary source we own.

## The efficiency leaderboard, read (topic 736678, 22:20Z)

The unexamined evidence from the section above, now looked at. Four competitor statements, all
self-reported, all unverifiable — but they are *statements about a public board* rather than about
private results, so they are cheaper to lie about and easier to check later.

1. **"As of Aug 22, not many of the public LB top 50 also appear in the Efficiency LB top 50 — about
   32%."** Read alone this argues FOR ensembling: two thirds of the high scorers are slow.
2. **"The top two Public LB submissions seem to be used for the Efficiency LB as well, so we should be
   careful not to accidentally get a high score through ensembling."**
3. **"I'm surprisingly #1 on the efficiency LB right now. No ensemble."**
4. A poster at **Efficiency 22nd / Public 78th with 0.926** on a ~20-minute pipeline: preprocessing plus
   a single model trained 5-fold.

**(1) and (2) are not in conflict, and the shape they make together is the reading.** The broad top-50
is slow, and the very top is fast. Ensembling is what the *plateau* is made of; the leaders are not
there because of it.

That is the third independent statement in two days that our gap is **member quality**, and this one is
the strongest of the three because it is about the board rather than about a self-reported number: we
arrived at 0.939 through six members, and the entries above us appear to be single models.

It does not license abandoning the ensemble — a six-member 0.939 is still our best result and diversity
is the only axis with a *measured* board transfer here (`knee-nas5`, 2.4×). It prices the two lanes
against each other: **another seating variation is a knob on a plateau the field says tops out near
0.94, and a better member is the axis the leaders are on.** That agrees, from the outside, with what
`interpretations/kneemri/2026-08-26_our-model-is-past-its-teacher.md` concluded from the inside — and
the two disagree about the mechanism, which is worth stating plainly rather than merging: the forum says
*better model*, our own gold-label reading says *better labels*. Both point away from seating.

### One technical asset, free

A competitor profiled inference for ~1300 studies and published the breakdown: header pass 0.5 min ·
**slice ordering 12.0 min** · decode + resize 4.1 min · container + import 5.0 min · GPU 5-fold 10.0 min.
The ordering pass opens the geometry tags of every slice in every selected series (~186 files a study)
to sort by projection onto the slice normal, then keeps 9. What they tried and what it bought:

- threads saturate at **64** — 128/256/512 change nothing, so it is not latency-bound;
- reading every Nth slice and interpolating drops ordering agreement to **~39%**, no better than chance;
- **`InstanceNumber` instead of the projection: 94.5% agreement, and ZERO speedup** — the cost is opening
  the file, not parsing the tag.

We have no runtime constraint and no efficiency-board entry, so none of this is actionable today. It is
worth keeping for the one case where it becomes load-bearing: if we ever ship a wide ensemble under the
9-hour limit, the ordering pass — not the GPU — is what will bind.

---

## Re-read 2026-09-16 10:14Z — the gap closed from our side, and the mechanism they name is one we never tested

Same tool, same topic, no slot. Three weeks on, the forum's single-model claims have barely moved while our
ensemble has: highest single claim is now **0.943** ("0.942 raw, 0.943 with TTA at 288x288"; separately
"0.934 @ 224px … Update: 0.943 with some tweaks, still @ 224px"), against 0.942 when this file was written.
Below it: 0.938 single fold · 0.937 CoAtNet 384 single fold · 0.936 small ResNet 224 · 0.930 in 46 minutes ·
0.929 and 0.926 five-fold on LLM output labels only · 0.92 · 0.915 DINOv2. Topic 737696 the same day has the
Raptor author at **0.936–0.938 with one un-blended model**.

Our pinned champion is **0.946** (`comp/champion.json`, `tripleseat_v2`, seven members). **Nothing visible on
that forum is above it** — which is the standing rule, now re-measured rather than remembered: an external
result is priced against our distance to the top, and a series that tops out below our champion is a
hypothesis, not advice.

**What still bites, for the third independent time.** Their ONE model reads 0.943 against our SEVEN-member
0.946. Three ticks is what the entire mix buys us over a rival's single part. The gap is per-member quality,
not mixing — stated here on 2026-08-26, in `research/2026-08-25_competitor-single-model-scores.md` the day
before, and again now with the spread unchanged.

**The mechanism they name is not the one this repo closed.** Unprompted and repeatedly: *"labels are still my
greatest lever, especially pseudo labeling and blending different labels"*, plus two entries trained on **LLM
output labels only** reaching 0.926–0.929 five-fold. That is NOT the label-correction axis 735304 closed
earlier (correcting thousands of verified extraction failures moved no LB). Correcting wrong labels and
*manufacturing more* labels are different mechanisms, and only the first was tested — here or there. Given
the supervision bottleneck this project has measured from the inside (58 of 4,407 studies), pseudo-labeling
is the untested axis with the largest claimed carry on the board.

**One correction to the record, made the same session.** A "0.949 from a single small ResNet" was quoted into
bead knee-3kpv9 against this topic and then promoted to a P1 (knee-k1d9v). It is not in the thread; the
nearest real rows are 0.936 ("small resnet model, 224x224") and a separate "0.930 in 46 minutes". Both beads
carry the withdrawal. A cited number nobody re-opens is indistinguishable from an invented one.
