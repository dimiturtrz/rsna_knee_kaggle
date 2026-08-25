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

**3. Aggregation.** Already closed from both ends by the ledger, before today, and closed harder than
this section first stated: 20 cached slices against 12 reads −0.0001 (`knee-ezz`), and **four** pooling
operators span 0.0019 on one recipe — elementwise max 0.8457, an order-aware gate 0.8461, unweighted
mean 0.8473, a trained permutation-blind gate 0.8476 (`attend_true`, `depth_true`). The order-aware one
matters to this page's argument and was missing from it: all three of the others are permutation-blind,
so slice ORDER was the clause `attend_true` left open, and `depth_true` closed it by landing
second-worst. Neither more inputs to the slice aggregator, nor a better rule for combining them, nor
telling it which slice came before which, moves the number.

Read the plumbing note with these: `attend_dense`'s gate sat in no optimiser group (knee-7gyw), so it
ran as a fixed mean. That is not a hole in the triple — it is why the MEAN point exists. `parameter_groups`
now selects the readout by EXCLUSION from the backbone (`model.py:489`), which is the form of the fix
that cannot lose the next sibling module the same way, and both trained-gate arms above are post-fix
re-runs with the gate's weights verified non-zero before their verdicts were written.

**4. The pooled vector's WIDTH — the hypothesis this page originally closed on, answered by two arms
already on disk.** `--pool cls_mean_focal` concatenates the mean of each channel's top eighth over the
patch grid, and its own flag help names this page's four structures as the reason it exists: a finding
occupying 1–2 of 256 patches is diluted by a plain mean, and that dilution ratio is invariant to
resolution and to slot coverage, which is exactly why (1) and (2) could come back flat while the
mechanism stayed alive. It is the best-argued remaining explanation, and it has been run twice:

```
focal_dense (20sl, vs band_dense)   focal +0.0001   regional -0.0114
focal_pool  (12sl, vs whole_stack)  focal -0.0004   diffuse  -0.0065
```

Both flat on the focal group, both paying on the diffuse one. And read against THIS page's six deficit
findings rather than against the tool's focal/regional split — which is a different partition, so the
structure-scale lines above do not answer it — `focal_dense` moves ACL +0.006, Lateral Meniscus +0.010,
Lateral OA +0.005 against MCL −0.008, Medial Meniscus −0.006, PF OA −0.003: **net ≈ +0.004 over the six,
and LATERAL-up / MEDIAL-down rather than deficit-up.** The tail does reach the finest structure in the
corpus, which is a real observation about where it spends attention; it does not reach the deficit SET.
No new run was needed to learn this and none should be queued — the axis was read per-label, and it is
the third mean-refutation this page has re-read at per-label resolution without reopening one.

**5. Supervision VOLUME — the first thing anyone reading the table above would propose, and it is
refuted by inversion at both ends.** If the deficit set were simply the findings the corpus has fewest
positive examples of, the whole page would be a data problem with a known lever. `core.data.labels.label_power`
on `report_labels_v2` gives each finding's positives in an 881-study fold, and set against the split:

```
DEFICIT side (model trails)              SURPLUS side (model beats its labels)
  Lateral Meniscus  -0.1925   224          Effusion         +0.1304   533   <- RICHEST
  ACL               -0.1225   245          Contusion        +0.0891   259
  MCL               -0.0964   153          Synovitis        +0.0508   116   <- 2nd poorest
  Medial Meniscus   -0.0938   469  <- 2nd richest
  PF OA             -0.0856   398  <- 3rd richest
  Lateral OA        -0.0793   217          Baker's          +0.0226   222
                                           Fracture         +0.0160    62   <- POOREST
  mean 284 positives                       mean 252 positives
```

The two POOREST labels in the corpus are both on the surplus side; the second- and third-richest are
both on the deficit side; and the deficit side carries MORE positives on average, not fewer. Prevalence
does not order this split — it very nearly anti-orders it.

Two things follow, and the second is a caution against over-reading the first. Positive COUNT is not
what distinguishes a finding the model beats its teacher on from one it trails on, so "get more studies"
is not licensed by this page, and MRNet (`knee-ombx`) should not be argued for on the strength of it.
But this does NOT say more data cannot help: it says volume is not the AXIS the deficit lies along, and
a corpus that differs in KIND rather than in count — one carrying explicit internal-derangement reads —
is untouched by the argument. It also does not disturb `knee-yrg`'s separate medial-over-lateral
prevalence question, which is about a gap WITHIN the deficit side, not about the split itself.

## Where that leaves the mechanism

The ledger's own conclusion from (3) was written days ago and is the one this page arrives at from a
different direction: **"the limit is upstream, in what one slice's feature vector contains."** Add
today's two refutations and the encoder-scale null from `base224` (independently reproduced by a
competitor in topic 735154), and the surviving statement is narrow and uncomfortable:

- it is not the trunk's **capacity** (small ≈ base, twice measured),
- it is not the **image's** resolution (336 ≈ 224 on the focal four),
- it is not the **input's** coverage (nine slots ≈ six, and worse on menisci),
- it is not the **aggregator** (slice count, pooling rule and slice order all flat),
- it is not the pooled vector's **width** (the focal tail, read per-label, ≈+0.004 over the six),
- it is not the **labels**, for these four findings — the reports describe them at 0.84–0.99,
- it is not supervision **volume** (the split anti-orders with positive count).

Everything cheap has been eliminated and the deficit is still −0.049. The list is now closed on every
DECODE-side axis this project can name: what is read (resolution, coverage), how much of it is read
(slice count), and how it is combined (operator, order, width). Two of those five were closed today by
new arms; the other three were already on disk and needed reading, not running. That is not a lever yet; it is a
well-localised open question, and stating it as one is the honest end of the page. The next thing it
licenses is diagnostic rather than corrective: find out **whether our own members share the published
members' deficit shape**, since every number here rests on assuming they do, and that costs no card —
only a set of predictions from a member fitted without annotation upweighting.

## The split turned out to be PREDICTIVE, which is the strongest thing on this page

Everything above is elimination. This is not, and it was earned after the page was written: `blendlabels`
— a training table blended from three independent readers of the same reports — ran the same evening
against `native224`, and the six findings it moved are the six this page had already named.

```
by structure scale:        focal +0.0000   regional +0.0083
its 6 label-bottlenecked:  mean  +0.0070   (Baker's, Contusion, Effusion, Fracture, Medial OA, Synovitis)
twelve-label MEAN:         +0.0034
```

`run_comparison --split` derives that six-finding set from the supervision gap itself, so it is not a
set this page handed it. The claim group moves at **twice** the twelve-label mean while the focal group
moves +0.0000 to four decimal places. A partition that was fitted to explain a past measurement and then
correctly says where an unrelated future change will land is doing more than describing — it is the one
result here that would survive someone disputing the -0.049.

The exception is recorded rather than smoothed: MCL +0.0239\* is the largest single delta and MCL is a
deficit-side structure a better reader should not reach. One marked crossing in twelve on the
third-poorest label (153 positives) is about what chance gives, and the other three deficit structures
behave — but it is not explained.

And it sizes the label lane honestly, which the elimination sections cannot: better readers buy about
+0.007 on the six findings they reach and about +0.003 on the metric. Both sit at or under the 0.0062
floor. **The lane is live and it is small** — an order of magnitude short of the 1.5% bar, so it is a
thing to keep, not a thing to plan around.

## And then something moved the deficit set — a corpus that differs in KIND

Every closure above is a DECODE closure: how the pixels are read, pooled, sized, ordered, covered. The
one axis that has ever cleared the bar is data VOLUME (`half_data`, −0.0234 on half the corpus), and the
prevalence check further up refuted volume as the *ordering* of the deficit while leaving exactly one
door open — a corpus differing in KIND rather than in count, one carrying explicit internal-derangement
reads. `merged_corpus` walks through it.

736 kneeMRI (Rijeka) studies, concatenated into our corpus as one cache and trained jointly. They are
sagittal PD fat-suppressed on one Siemens Avanto — `SLOTS[0]`'s predicate exactly — and their ACL labels
are expert double-blind reads OF THE IMAGES, not text mined from a report. They carry one label column.

    deficit six                       surplus six
    Lateral Meniscus  +0.0196*        Synovitis   -0.0206*
    ACL               +0.0161         Contusion   +0.0035
    MCL               +0.0062         Medial OA   +0.0023
    Lateral OA        +0.0016         Effusion    +0.0017
    Medial Meniscus   -0.0024         Baker's     +0.0004
    PF OA             -0.0044         Fracture    -0.0005
    mean              +0.0061         mean        -0.0022

    twelve-label mean +0.0020    focal +0.0056    regional -0.0031

**The twelve-label mean is a third of the floor and says nothing.** The split is the result: the two
halves separate by 0.0083 in the pre-registered direction, and three of the four internal-derangement
structures move up.

**The line that carries the page is Lateral Meniscus.** Rijeka supervises the ACL and nothing else —
`Targets.pair` weights an absent column at 0.0, so there is no meniscus gradient anywhere in those 736
studies. Yet Lateral Meniscus is the largest mover and one of only two deltas clearing its own interval.
That gain cannot be supervision. It has to be the encoder: expert focal reads taught it to encode focal
structure, and the benefit landed on a structure the new labels never named. Which is precisely the
hypothesis every decode closure above leaves standing — the limit is *what one slice's feature vector
contains*, and this is the first thing to change it.

Held honestly: one arm, one seed, a wide paired floor (±0.0400; the control predates per-study
predictions), and **Synovitis −0.0206\* is a starred cost on a surplus finding that is not explained.**
Nothing here clears 1.5%. What it does is re-point the schedule: 21% more studies with ONE label column
moved the set that six decode axes could not touch, so the next lever is MORE OF THIS KIND, not another
knob. MRNet (`knee-ombx`, owner-blocked on the RUA) is 1,370 studies with image-read ACL **and meniscus**
labels — 1.9× this arm on the axis that just moved, supervising the structure that moved most. That case
is now made on KIND. It was never available on volume, and the prevalence refutation above says so.

## WHOSE MODEL IS THIS A PAGE ABOUT? Not ours, on the part that matters most

Every ACHIEVED number above comes from the published members, because they are the only predictions in
this project that never saw the annotations. The page therefore measures THEIR model against the labels,
and the whole time it has been read as a statement about ours. It is not one, and the difference is
largest exactly where the page's argument is loudest.

Our best single member (`lr4e-5_ws`) against the published ensemble, per label, on the 909 studies both
hold out, scored against `report_labels_v2`:

    Lateral Meniscus  +0.0661        Contusion    +0.0165        MCL         +0.0020
    Medial Meniscus   +0.0451        ACL          +0.0128        Fracture    -0.0017
    Baker's           +0.0220        PF OA        +0.0130        Synovitis   -0.0125
    Lateral OA        +0.0102        Medial OA    +0.0085        Effusion    +0.0091
    mean ours 0.8525   published 0.8366   ours-published +0.0159, ahead on ten of twelve

The two largest gaps in the whole table are the two MENISCI — the deficit six's biggest members, where the
published model reads −0.1925 and −0.0938. Our own weakest findings are MCL (0.7908), Fracture (0.8066),
Lateral OA (0.8178) and PF OA (0.8392); our menisci and ACL are among our STRONGEST. The 4-of-4
internal-derangement shape, which is the page's headline, does not hold for our model: two of those four
are in our top three.

WHAT THIS DOES AND DOES NOT LICENSE. It is a different RULER — report labels over 909 studies, not
radiologist truth over 58 — so it cannot say the deficit is closed, only that the model the deficit was
measured on is not the model we ship. The right instrument cannot be run: `supervision_gap` needs
predictions that never trained on the annotations AND that cover them out of fold, and of our 58
annotated studies only 13 fall in fold 0 while `member_fold1..4` left no prediction file at all — union
coverage 2 of 58. Producing it means training four folds, which are banned. So the honest position is:
the −0.049 macro and the 4-of-4 shape are facts about the published members, must not be quoted as
facts about ours, and there is no cheap experiment that converts one into the other.

The MRNet case does NOT rest on this. `merged_corpus`'s deficit/surplus split above was measured on OUR
members under OUR ruler and stands on its own evidence; it is the KIND argument, and it survives this
section untouched.

## What it does NOT say

It does not say ensembling is worthless or that the roster should change; those axes closed separately
today. It does not say the label lane is closed — six findings ARE reader-bottlenecked, Effusion by
+0.1304, and `severity_boundary` shows separately why the Effusion half of that is not recoverable by
parsing reports better. And it does not price the −0.049 as available: a deficit measured is not a gain
banked, and nothing on this page has been shown to close any part of it.

Beads: `knee-487l` (this page), `knee-2dk1` (the bottleneck reading), `knee-uj15` (the single-model frontier).
Research: `research/2026-08-25_competitor-single-model-scores.md`.
