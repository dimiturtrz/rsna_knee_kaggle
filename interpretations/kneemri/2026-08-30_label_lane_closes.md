# The label lane closes at +0.0003, and directional gains did not compose

**2026-08-30 · card-free, arithmetic over cached holdout arrays · knee-fuhq**

## What was being asked

knee-fuhq measured a real thing: our `report_labels_v2.csv` targets score **0.8705** against the 58
radiologist-annotated studies, our best members read ~0.873 against those same targets, and a published
model beats OUR LABELS on ten of twelve findings. Label noise, not architecture, looked like the ceiling.
The proposed treatment was the standard one — average a second independent LLM reading of the same reports
into the targets — and `blendlabels_s2026` is the member trained that way. It read **+0.0034** on the mean
and **+0.0070** on exactly the six findings where the labels are the ceiling, while `seat_value` priced it
at **+0.0001** marginal.

That is not a contradiction: `seat_value` prices a MEAN contribution, so a member that helps on half the
findings and dilutes on the other half reads as nothing. The open question was whether **per-label routing**
— the instrument that does not average the split away — recovers the +0.0070.

## The estimator, which is the whole point

`kneemri/evaluation/label_combine.py` carries its own matched control. Selecting the top-`k` members per
label at `k = every member` **IS** the flat rank-average, so one `k` curve holds candidate and baseline under
the SAME split-half estimator. Comparing a split-half number against a whole-holdout one would compare two
rulers. Both `_cf` members are dropped from both arms — their fold −1 holdouts sit inside their own training
set and would inflate a split-half estimate.

## The numbers

| arm | split-half peak | flat control (`k` = all) | routing gain |
|---|---|---|---|
| honest12's ten fold-0 members | **0.8751** (k=6) | 0.8742 | +0.0009 |
| the same ten **+ blendlabels_s2026** | **0.8754** (k=6, 8) | 0.8745 | +0.0009 |

In-sample peaks are 0.8790 and 0.8791 — the ~0.005 gap to split-half is the selection optimism the module
exists to strip, and it is exactly the size the module warned about (twelve choices on 881 studies, as few
as 62 Fracture positives).

## Three readings

**The label lane closes.** knee-fuhq pre-registered "flat or lower means the +0.0070 was inside the
selection optimism, and the label lane closes for good rather than being demoted." blendlabels is worth
**+0.0003** peak and **+0.0003** flat — 1/50th of the significance bar, and independently consistent with
`seat_value`'s +0.0001. Averaging a second LLM reading into the targets does not convert the label/model
split into score.

**knee-orjq is answered on a free instance: directional gains did NOT add.** A +0.0034 mean gain and a
+0.0070 six-finding subset gain composed into +0.0003 under an honest estimator. This is one instance, not a
law, but it is the first actual test and it came back negative.

**Per-label routing is not a lever either.** Routing buys +0.0009 over the flat rank-average in BOTH arms —
the same order as knee-m69u's +0.0002 per-label weighting ORACLE headroom, which bounds every weighting
scheme. The combine axis is where it was.

## What is not closed

The label-noise SPLIT itself stands as a description of the corpus: the labels are the ceiling on Effusion
(+0.1304), Contusion (+0.0891), Synovitis (+0.0508), Medial OA (+0.0419), Baker's (+0.0226) and Fracture
(+0.0160), and the model is the ceiling on the other six. What is refuted is that re-labelling monetises it.

With the knob axes closed (knee-m69u) and this lane closed, what remains that could plausibly clear 1.5% is
**trunk diversity** — see knee-spun: 84% of the scored pool is one encoder, and every closed axis above was
measured inside that monoculture.
