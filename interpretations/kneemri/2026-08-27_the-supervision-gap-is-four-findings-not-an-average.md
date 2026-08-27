# The supervision gap's aggregate spans zero; four of its twelve findings do not

2026-08-27, evening. Source: `python -m kneemri.evaluation.supervision_gap --labels report_labels_v2.csv`,
run with the paired band this session added to it. 58 radiologist-annotated studies, ~696 cells. No card
was used — this is CPU seconds against arrays that already existed.

## The number that was steering the queue, now bounded

`2026-08-25_forum-bottleneck-vs-our-own-diagnostic.md` reported our labels at gold macro AUC 0.8658
against our model at 0.8392 and read the 0.027 between them as **"our image model sits below its own
teacher"** — the modelling-headroom branch. It also stated its own weakness plainly: 58 studies, no
interval computed for any number on the page, so the 0.027 was separated from zero by nothing.

Bounded, paired over studies, it reads:

```
labels - model: +0.0266  [-0.0082, +0.0610]  over 58 studies
```

**The interval spans zero.** The aggregate cannot say which half of the pipeline is weaker. The headline
claim does not survive its own measurement, and the document that made it had already named the reason.

Paired is the only honest ruler here: both columns are scored on the same 58 studies against the same
annotations, so per-study difficulty cancels in the difference. Two unpaired `mean_interval` bands on 58
rows would be far wider, overlap almost entirely, and answer a different question.

## What DOES survive, and it is the more useful half

Four of the twelve per-finding bands exclude zero:

```
  ACL                -0.1225   model side, resolved
  MCL                -0.0964   model side, resolved
  Lateral Meniscus   -0.1925   model side, resolved
  Effusion           +0.1304   label side, resolved
```

Twelve bands read together are twelve chances to exclude zero, so four is a shape of the evidence, not
four independent findings. But the four are the four LARGEST gaps on the table, and they land on both
sides — three where the model fails to extract signal the report carries, one where the model corrects a
reader that cannot see the finding. That is not the pattern chance produces from a flat truth.

**So the split survives while the average dies.** Which is exactly what `Bottleneck`'s own docstring
argued before anything measured it: *"Averaged over all twelve, either change is diluted by the six it
cannot affect — six findings of noise added to six of signal."* The aggregate spanning zero is that
dilution, measured. The module was right about its own aggregate and the interpretation read past it.

It also converges with the `blendlabels` arm, which pre-registered the six label-bottlenecked findings
from this split and landed +0.0070 on them against +0.0034 on all twelve. The partition was predictive
there and it is resolved here, by two unrelated routes.

## What this changes about the queue

Neither branch is licensed on the aggregate any more — not the label-bottleneck reading the forum
converged on, and not the modelling-headroom reading we preferred. Both were arguments about a number
that spans zero.

What is licensed is narrower and better aimed. **The three resolved deficit findings are ACL, MCL and
Lateral Meniscus** — thin, ligamentous and fibrocartilaginous structures whose report descriptions are
excellent (0.9865, 0.9762, 0.8422) and whose pixels the model cannot read (0.8640, 0.8798, 0.6497). No
label work can reach them. That is a resolution and capacity question, and it is the first version of the
capacity argument this project has held that rests on a bounded number rather than on a mean.

**The one resolved label-side finding is Effusion**, which is precisely what
`2026-08-25_effusion-threshold-ruler.md` measured a defect in that same day — and which that document
correctly sized as a free rider on a label arm rather than an arm of its own (509 rows, 11.5% of the
corpus, one of twelve findings, target moves ≤ 0.18). Nothing here upgrades it; it confirms the finding
is real and leaves the sizing alone.

## What this does NOT say

It does not say the model is not below its teacher. It says 58 studies cannot tell, which is a statement
about the ruler and not about the model. A wider annotated set would answer it, and the annotations are
the host's, not ours to make.

It does not license reading the four resolved findings as four findings. One aggregate comparison was
named in advance; the twelve per-finding bands were not, and this document is the only place they appear.

It does not price an arm. The deficit structures being real does not say what fixes them, and the honest
next step is the cheapest instrument that separates resolution from capacity on those three specifically,
not a bigger backbone armed on the strength of this page.

## The first instrument it licensed, run the same hour: it is NOT in-plane resolution

The section above asks for the cheapest thing that separates resolution from capacity on ACL, MCL and
Lateral Meniscus. That instrument cost no card and no training, because the pair already exists on disk:
`lr4e-5_ws_s2026` (336 px) against `res392_ws_s2026` (392 px, 448 cache), identical on every other flag —
`run_comparison` prints the diff itself as `image_size 336→448, train_size 0→392`.

The ledger had already read this pair, twice, at two encoder rates, and called it flat: +0.0002 and
−0.0002 against a 0.0062 floor. **Both readings are MEANS**, and this page's whole content is that the
mean dilutes. So the question was live in a way the ledger could not answer: does the pixel axis pay on
the three thin structures even though it is flat over twelve?

Pre-registered before running, from `band_dense` being the axis that moved collaterals: flat on the three
as well. It is.

```
ACL +0.0109   MCL -0.0088   Lateral Meniscus -0.0001      MEAN -0.0002
```

Mixed signs, no delta marked against its own paired band (half-widths to ±0.0353), and the tool flags
the pair as INDICATIVE because one run's predictions were re-derived. Nothing resolves. `run_comparison`
also prints its own structure-scale summary as **INVERTED** — focal −0.0027 against regional +0.0032,
*"the tighter view helped exactly the findings it should have cost"* — which is the opposite of what a
real resolution effect looks like.

**So the deficit is not pixels.** That properly retires `knee-jk3`'s *"336 px for findings a few voxels
across"*, which the ledger had already read down at the mean and which is now read down on precisely the
findings it was a claim about. And it is worth more than a null usually is, because it would otherwise
have been tomorrow's first card arm: the capacity argument this page licensed points at through-plane
coverage and architecture, not in-plane sampling — consistent with `band_dense` (20 slices, +0.0053)
being the one geometry change that ever moved this family.

## The instrument

`SupervisionGap.separation` now prints the band beside the aggregate, and the CLI calls it unconditionally
— the number that steers the queue cannot be printed bare again. It is `LabelScores.paired_mean_interval`
(built this session) plus `paired_interval` for the twelve, and the binarisations were checked to match
(`> MIDPOINT` and `> PRESENT`, both 0.5) before any number here was believed.

Related: `2026-08-25_forum-bottleneck-vs-our-own-diagnostic.md` (the claim this bounds),
`2026-08-25_effusion-threshold-ruler.md`, `2026-08-27_every-no-card-axis-is-priced-and-flat.md`,
`knee-w7xa` (the blendlabels partition).
