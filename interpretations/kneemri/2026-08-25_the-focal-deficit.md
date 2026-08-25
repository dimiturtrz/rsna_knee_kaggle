# The four internal-derangement structures are model-bottlenecked, and three explanations are dead

2026-08-25, ~18:50Z. All figures re-run live rather than quoted from prose. No card, no submission slot.

## The measurement

A competitor in discussion topic 735826 proposed a diagnostic:

> *"Check to see how your label extractor performs relative to the 58 provided labels. Then check how
> your image model performs on them. If your image model is not beating the label extractor 'teacher',
> there is likely modeling headroom. If your image model is beating the label extractor by a large
> margin, the quality of the labels might be holding the image model back."*

`kneemri.evaluation.supervision_gap` is that diagnostic, and it has been in this repo since 2026-08-14
with more resolution than the thread asks for — it splits **per finding** rather than in aggregate.
Run against `report_labels_v2.csv` and the published out-of-fold members:

```
58 annotated studies with a held-out prediction
aggregate: labels 0.8658   model 0.8392
finding                labels    model  model-labels
Effusion               0.8149   0.9453       +0.1304
Contusion              0.7530   0.8421       +0.0891
Synovitis              0.6870   0.7378       +0.0508
Medial OA              0.9085   0.9504       +0.0419
Baker's                0.9321   0.9547       +0.0226
Fracture               0.8701   0.8861       +0.0160
  --- above: the model beats its own labels, so the READER is the weak part ---
Lateral OA             0.7892   0.7099       -0.0793
PF OA                  0.8861   0.8005       -0.0856
Medial Meniscus        0.9435   0.8498       -0.0938
MCL                    0.9762   0.8798       -0.0964
ACL                    0.9865   0.8640       -0.1225
Lateral Meniscus       0.8422   0.6497       -0.1925
```

**In aggregate our model does not beat its teacher** (0.8392 against 0.8658). By the thread's own rule
that reads as modelling headroom, not a label ceiling — which is the opposite of the poll's consensus
answer for that thread, and it is our number rather than theirs.

**The split is not random.** All four internal-derangement structures — ACL, MCL, medial meniscus,
lateral meniscus — sit on the deficit side, **four of four**, and they carry four of the five highest
report-label AUCs on the board. The report names an ACL tear explicitly and the extractor reads it at
0.9865; the model reaches 0.8640 from the pixels. The six deficit findings sum to **−0.049 macro**,
three times the significance bar.

**Denominators, before anything is built on this.** 58 annotated studies, 696 cells; `label_power` puts
the per-label standard error here at 0.04–0.09, so **no single row above is a finding** and the −0.1925
on lateral meniscus is not separable from the −0.0793 on lateral OA. What is claimed is the STRUCTURE:
six above, six below, and the four internal-derangement structures all below. Second caveat, and it is
the one that bounds the whole page: these are the **published** members, because ours are fitted with the
annotated studies upweighted and scoring them here would be scoring training data. So this prices the
public baseline's deficit and assumes ours shares its shape. That assumption is untested.

## Three explanations, tested and dead

Having a deficit localised to four structures invites a mechanism. Three offered themselves, and the
suspicion behind all three was the same: *a 12-label mean would average a four-label effect away* —
which is a failure mode this project has a written rule about. Tested; the rule did not apply.

**1. In-plane resolution.** Focal structures are small; maybe 224 px destroys them. `run_comparison` on
the matched single-flag pair (`lr4e-5_ws` at 336 vs `native224`, `image_size 336→224`):

```
ACL +0.0062  MCL -0.0037  Medial Meniscus -0.0072  Lateral Meniscus -0.0169
by structure scale: focal -0.0035  regional -0.0007
  -> REFUTED — nothing gained at either scale, so resolution was not the bottleneck
```

336 buys about +0.005 on the focal four against this pair's own ±0.0332. Not a resolution problem, and
the tool that says so had the structure-scale grouping built in already.

**2. Missing sequences.** `slots.py` records that 2,421 of 24,371 series (9.9%) route to no canonical
slot — 83,297 slices unseen, among them coronal PD without fat suppression, *a standard meniscus
sequence*. A discarded meniscus sequence against a meniscus deficit is as clean a story as this page was
going to get. The ledger closed `extended_slots` as "refuted **on the mean**" (+0.0002), which is exactly
the phrasing that should invite a per-label re-read. Re-read:

```
extended_ws vs wholestack (scheme canonical→extended, +1,139 coronal series)
Lateral Meniscus -0.0130   Medial Meniscus -0.0095   ACL -0.0055   MCL +0.0056
by structure scale: focal -0.0029  regional +0.0045
```

The menisci move the **wrong way**. The predicted signature is absent, not averaged away. The row was
right and the re-read confirms it rather than reopening it.

**3. Aggregation.** Already closed from both ends by the ledger, before today: 20 cached slices against
12 reads −0.0001 (`knee-ezz`), and three pooling operators — elementwise max, unweighted mean, a trained
64-wide gate — span 0.0019 across `attend_true` / `attend_dense`. Neither more inputs to the slice
aggregator nor a better rule for combining them moves the number.

## Where that leaves the mechanism

The ledger's own conclusion from (3) was written days ago and is the one this page arrives at from a
different direction: **"the limit is upstream, in what one slice's feature vector contains."** Add
today's two refutations and the encoder-scale null from `base224` (independently reproduced by a
competitor in topic 735154), and the surviving statement is narrow and uncomfortable:

- it is not the trunk's **capacity** (small ≈ base, twice measured),
- it is not the **image's** resolution (336 ≈ 224 on the focal four),
- it is not the **input's** coverage (nine slots ≈ six, and worse on menisci),
- it is not the **aggregator** (slice count and pooling rule both flat),
- it is not the **labels**, for these four findings — the reports describe them at 0.84–0.99.

Everything cheap has been eliminated and the deficit is still −0.049. That is not a lever yet; it is a
well-localised open question, and stating it as one is the honest end of the page. The next thing it
licenses is diagnostic rather than corrective: find out **whether our own members share the published
members' deficit shape**, since every number here rests on assuming they do, and that costs no card —
only a set of predictions from a member fitted without annotation upweighting.

## What it does NOT say

It does not say ensembling is worthless or that the roster should change; those axes closed separately
today. It does not say the label lane is closed — six findings ARE reader-bottlenecked, Effusion by
+0.1304, and `severity_boundary` shows separately why the Effusion half of that is not recoverable by
parsing reports better. And it does not price the −0.049 as available: a deficit measured is not a gain
banked, and nothing on this page has been shown to close any part of it.

Beads: `knee-487l` (this page), `knee-2dk1` (the bottleneck reading), `knee-uj15` (the single-model frontier).
Research: `research/2026-08-25_competitor-single-model-scores.md`.
