# Labels or pixels: which half is holding each finding back

*2026-08-14, afternoon. Every number here is CPU arithmetic over predictions and headers that already
existed. No GPU, no submission.*

The morning established a corpus-sized ruler and found the label profile was not the one the plan was
built on ([the ruler was five times too small](2026-08-14_the-ruler-was-five-times-too-small.md)). This
is what that ruler then said about **where** the model is losing, and the design that follows from it.

## The split

Our labels are an LLM reading reports; the competition scores radiologists reading images. Where the
report says YES the radiologist agrees **65.3%** of the time — a third of the positives we train on are
not positives. That says the labels are noisy. It does **not** say what to do, because the answer
differs by finding.

Two numbers per finding against the same radiologist truth, on the 58 annotated studies, using the
published members' out-of-fold predictions — the only predictions that never trained on those
annotations (`python -m kneemri.evaluation.supervision_gap`):

```
aggregate: labels 0.8639   model 0.8391

Effusion          0.8137 → 0.9453   +0.1317
Contusion         0.7470 → 0.8421   +0.0951
Medial OA         0.9085 → 0.9504   +0.0419
Synovitis         0.6953 → 0.7372   +0.0418
Fracture          0.8549 → 0.8861   +0.0312
Baker's           0.9330 → 0.9547   +0.0217
--- above: the model beats its own labels, so the READER is the weak part ---
PF OA             0.8790 → 0.8005   −0.0785
Lateral OA        0.7892 → 0.7099   −0.0793
Medial Meniscus   0.9435 → 0.8498   −0.0938
MCL               0.9762 → 0.8798   −0.0964
ACL               0.9841 → 0.8640   −0.1201
Lateral Meniscus  0.8429 → 0.6497   −0.1932
```

**"Ceiling" is the wrong word for the label column and the module refuses to use it.** A model fitted on
noisy labels can beat them by looking at the image, and here it does on six of twelve. Effusion's label
predicts the radiologist at 0.81 and the model reaches 0.95 — it is correcting the reader.

The split assigns different work to different findings. A better label reader (`knee-dbw`) can only pay
above the line. Below it the report describes the finding well and the model cannot reproduce it, so the
signal is in the label and the pixels are where it is being lost.

Caveat that limits every row: n=58, 9–35 positives per label, so per-label intervals are wide. The split
and the two largest gaps are the evidence; no single row should be planned around alone.

## The below-the-line group is sorted by structure size

Lateral Meniscus, ACL, MCL, Medial Meniscus — small, thin, spatially continuous. Above the line:
effusion, Baker's cyst, contusion, osteoarthritis — large, diffuse, fluid.

That is not a fact about labels. It is the architecture saying it cannot resolve small structures.

**ACL is the informative one.** It is a large central structure, the report describes it almost
perfectly at 0.9841, and the model loses 0.12 to it. Size cannot be the explanation there, so pure
resolution is not the mechanism for ACL and something about how depth is sampled is the candidate that
fits (`knee-ghd`).

## What was tested and did not survive

**Discarding depth.** The cache keeps 12 slices of a median 36 acquired, so two thirds of the imaging is
gone before the model sees anything — one of the four causes `knee-jk3` names for the 0.054 frontier
gap, and never tested. If it cost accuracy, the pixel-limited findings should score worse where more was
discarded. Across three bands of 1,469 studies by acquired depth they are flat or drift the **other**
way; 2 of 12 intervals exclude zero against a chance expectation of 0.6, and one of those points against
the hypothesis (`python -m kneemri.evaluation.stack_coverage`).

Not support for the discard cause. Stated as an association and not more — deeper stacks travel with
scanner, protocol and habitus, so a real cost could be masked by deeper scans simply being better scans.
The honest test is to cache 24 slices and retrain.

*The module carries its own correction: the first version reported bare deltas, and at 300 resamples the
Lateral Meniscus bound moved 0.0063 across resample seeds — enough to flip whether it cleared zero, a
verdict decided by a random seed. At 1500 the spread is 0.0015. An interval whose own Monte Carlo error
exceeds the effect is not an interval.*

## What the geometry says instead

- field of view **160 mm** median; our crop is **130 mm** — it removes 19% of the linear extent, so two
  thirds of every image is outside the joint
- at 336 px that is **0.387 mm/px**; a meniscal tear is 1–2 mm, and DINOv2 patches at 14 px, so **a tear
  is a fraction of one token**
- an 80 mm joint-centred crop at the same 336 px is **0.238 mm/px — 1.6× finer for identical compute**
- the crop is centred on the **image**, and the tissue centroid sits a median 7 mm, p90 18 mm, max 32 mm
  away from it

Both directions have already paid separately: `res392` (finer) is our best single member at 0.8523, and
`whole_stack` (more context) is our biggest arm at +0.0147. A single scale must trade one against the
other. That is the argument for **multi-scale reading** (`knee-1nn`) rather than for picking a scale.

## The order this has to happen in

The 15 mm of margin a 130 mm crop leaves absorbs an 18 mm positioning error for all but the worst tenth
of studies. An 80 mm crop has a 40 mm half-extent against a joint some 70–80 mm wide and absorbs
nothing — so a tighter crop centred on the image would test *"a tight crop that sometimes misses the
joint"* and fail for a reason unrelated to resolution. Centring therefore blocks the tight crop
(`knee-yrf` → `knee-1nn`), and both arms are staged with that coupling stated in their headers.

## What the plumbing audit turned up on the way

The 80 mm arm could not be launched at all, and forcing it would have been wrong in five places — the
cache key would have **overwritten the 99 GB every member was fitted on**, the trainer could not select
a crop, the member would have recorded 130 mm for an 80 mm run, and the serve path would have fed it
130 mm pixels. Every fingerprint would still have passed, because each is checked against its own
reading. The arm would have returned *"tighter crops do not help"*.

That is the recurring shape in this project rather than a new one: **a value recorded in a config while
the consumer reads a constant.** It is worth counting how often the cheap check beat the careful one
today — a CSV read killed the tie-break that a 4,407-study census had left standing, and the reading
rules that two beads argued over for three days were written in plain text inside checkpoints nobody had
opened.
