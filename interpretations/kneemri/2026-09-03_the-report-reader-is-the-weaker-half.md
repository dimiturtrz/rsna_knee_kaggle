# The report reader is the weaker half, and it is weak on four named findings

2026-09-03. Two gold reads on the 58 radiologist-annotated studies, both taken on CPU because the card
is the owner's. Neither cost a slot. Both were available four days ago and neither had been taken.

## What was measured

`recal_dense_s2026` is `band_dense_hold`'s recipe with one variable changed: the report reader's six
confidence levels replaced by their measured positive rates (`report_labels_v3.csv`, fitted by
`kneemri.evaluation.level_calibration` on these 58 studies). The arm ran on 2026-08-30, entered the
ledger at proxy 0.8610, was rostered, and its **deciding** number — the one its own header
pre-registered, on gold, first — was never taken. Every gate in the repo was green over that hole.

    kneemri.evaluation.annotated_readings --device cpu
      recal_dense_s2026      0.9128
      band_dense_hold_s2026  0.8956
      paired  +0.0172  [+0.0004, +0.0343]  over studies

Gold up by more than the 0.0062 floor, proxy up by +0.0037 and so not down by the floor. That is the
arm's REPLICATES branch verbatim: the re-map is a property of the **table**, not of the 224 px corner
`recal_holdannot` measured it at. Same sign at two corners, larger at the second (+0.0105 → +0.0172).

## What it is not

`report_labels_v3.csv` was fitted on these same 58 studies. `--hold-annotations` kept the arm from
*training* on those rows — which is the defect `knee-9198` guards — but it does not make the gold ruler
independent of the treatment. `annotated_readings` says as much in its own docstring: a calibrated arm
read on the ruler its constants came from is a consistency check and never a confirmation.

The shared origin inflates **both** corners equally, so the corner-independence conclusion — the only
question this arm was built to ask — survives it. The absolute +0.0172 does not. The lower bound is
+0.0004 on n=58, which is about one study wide. Directional.

## The reading that matters more

The header also prescribed `supervision_gap`, which asks a different question: against the radiologist,
how does the **report label** score as a predictor, versus the **model fitted on those labels**?

    labels - model:  -0.0470  [-0.0717, -0.0218]   over 58 studies, 696 cells

The interval excludes zero. The model beats its own supervision. Per finding, where the model is ahead:

    Synovitis         0.6870 → 0.7945   +0.1075
    Fracture          0.8701 → 0.9417   +0.0715
    Lateral Meniscus  0.8422 → 0.8944   +0.0522
    Baker's           0.9321 → 0.9783   +0.0462
    Lateral OA        0.7892 → 0.8201   +0.0309
    Medial OA         0.9085 → 0.9349   +0.0264
    …
    MCL               0.9762 → 0.9070   -0.0692

MCL is the only inversion. Four of twelve per-finding bands exclude zero (MCL, Effusion, Baker's,
Contusion) — twelve reads, so that is a shape rather than four findings.

Three of the top four — Synovitis, Fracture, and the near-miss Baker's — are exactly the three
extreme-UNK labels named in `certainty_ramp`'s roster note and in `knee-fxgm`: Synovitis 84.2% unknown,
Fracture 56.4%, Baker's 45.9%. The label lane's live clause was written as a suspicion about thin
supervision. This is that suspicion with a denominator: **696 cells, 34.5% radiologist-positive, and the
report reader loses to the model by 0.047 on them.**

Where the report is *better* — MCL, and ACL and Medial Meniscus where the gap is under 0.004 — no label
work can help, because there is nothing left in the label to recover.

## What it licenses, and what it does not

REPLICATES says: price a roster rebuild on v3 and spend a slot. Three constraints bind that.

The rebuild is card work — every seat retrains — and the card is the owner's. Nothing starts until it is
released.

**It does not license a per-finding version of the same table, and the first thing this reading suggested
was exactly that.** `level_calibration`'s own docstring already argues it down, with the arithmetic: 58
studies give a finding at most 58 cells spread over five levels, so a per-finding map fits noise, where
pooled it is five parameters on 684 cells. It also states the ordering is the good part — mean per-finding
rank AUC 0.866 — so a per-finding correction would be correcting what works.

The two readings do not conflict; putting them together is what says where the lever is. `supervision_gap`
scores the label as a *predictor*, by AUC, which is ORDERING. `level_calibration` fixes SCALE and says
plainly that it "does not improve the reader and it cannot: the ordering is what a better reader would
move." So the four findings where the model beats its labels are four findings where the report reader's
**ordering** is bad — Synovitis at 0.687 is the clearest — and re-mapping levels, at any granularity,
cannot touch that. The +0.0172 is the scale fix landing, and it is not the lever.

The next thing to reach for is a better **extraction** on those four findings, and `report_reader`'s
docstring closes most of that too, also with the arithmetic, also measured before a card hour was spent.
On the UNK cells the report does not name the structure at all: Synovitis 1.8% of 1,424, Effusion 0 of
84. The silence is the REPORT's, not the reader's — a knee report discusses what was asked about. Only
Baker's (30.8% of 765) and Contusion (16.1% of 342) have text left on the floor, and that is 1-3% of the
label matrix moving one rung, which `report_reader` already declines to arm as a full sweep.

So of the four findings where the model beats its labels, two are unreachable by reading harder. What is
left on those two is not a label-quality problem at all, and it is the sharper reading of this whole
document: **for Synovitis and Fracture the report carries no signal and the image does.** 84.2% and 56.4%
of those cells are the constant 0.28 — supervision that asserts nothing — and the model reads 0.79 and
0.94 on them anyway. A label AUC of 0.687 against a model's 0.794 is not a mis-read label being beaten;
it is a model exceeding a *ceiling made of constants*, on cells where the supervision was never a
measurement.

That points at the UNK cells as a semi-supervised target rather than a reading target — `Targets` already
pulls them weakly by confidence, and the question nobody has asked is whether pulling them by the model's
own read does better. That is a real mechanism and it is unpriced. It is not claimed here as a lever; it
is the one direction this reading leaves open, and it needs its own pre-registration.

The slot cannot be spent yet. There is no v3-rebuilt member in the pool, so a submission today would be a
reseating over the existing pool, and that axis is closed five times over (`lensseat_v1`, `aucvote_v1`,
`monotone_v1`, `holdswap_v1`, `greedyseat_v1`). The slot waits for the member, not the other way round.

## The process finding

The arm's deciding number lived only in its header's prose, took two commands nobody ran, and sat unread
for four days behind a full green board. `knee-v78o1` filed that hole; this is the first time it has been
paid off. The check that would catch the next one asserts that an arm which pre-registers a branch has a
recorded branch selection — not that it ran.

Related: `2026-08-27_noise-binding-the-label-lane-is-live.md` (the over-call binding this sits beside),
beads `knee-qxyj`, `knee-v78o1`, `knee-fxgm`, `knee-mtqgg`.
