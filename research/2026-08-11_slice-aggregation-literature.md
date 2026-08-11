# What the knee-MRI literature does with the slice axis

Gathered 2026-08-11, after whole-stack reading turned out to be our largest single change
(+0.0147). The question this answers: having established that the slice axis carries signal we were
discarding, what does the field do with that axis, and what transfers?

## The frontier is not reachable by reading, because the competition is live

RSNA's knee challenge runs to 2026-10-22 with winners in November, so **no top-solution write-ups
exist**. The house rule — gather SOTA, replicate, then build past — has nothing to gather here. What
*is* available is the academic literature on the same task, which is a different and older body of
work built mostly on **MRNet** (Stanford, 1,370 exams, three labels) rather than on this competition's
4,407 studies, twelve labels, sixteen sites and twelve report languages.

Treat everything below as mechanism to steal, never as numbers to compare against: MRNet's task is
easier and its numbers are not commensurable with ours.

## What they do with the slice axis

| approach | slice aggregation | what it implies for us |
|---|---|---|
| **MRNet** (PLOS Med 2018) | per-slice CNN features, **elementwise max over ALL slices**, then FC | our three-slice window was the deviation; whole-stack replicates the canonical operation |
| **Medical Slice Transformer** (Sci Rep 2025) | **DINOv2 per slice** + a *Slice Transformer* attending across slices, plus within-slice patch attention | same backbone we run; attention over the axis we currently `amax` |
| two-stage transformer | ViT encoder per slice, sequence encoder with a `[CLS]` over slices | a study-level token instead of a pooled vector |
| **Memory-Guided Transformer** (Pattern Recognition 2025) | group attention across anatomical planes + cross-attention in a memory matrix | plane-level structure, which is our slot axis |
| **KneeXNet** | graph convolutions over anatomy, multi-scale fusion, contrastive pretraining | heavier; the contrastive part is the interesting half |

## What this changes about our plan

**Confirmed, already acted on.** Max over all slices is what the field does. We arrived at it from our
own depth-profile measurement and the prior art agrees — which is the cheapest kind of confirmation.

**The next step is named for us.** Every 2025 entry replaces the max with *attention over the slice
sequence*. Our `slot_features` already computes per-slice features and ends with `.amax(dim=2)`; the
slice axis is sitting there, and `SlotHead` already does query attention one level up over slots.
Filed as `knee-5ag`.

**A counter-signal we should not ignore.** MRNet reaches ~0.97 on ACL with **AlexNet at 256 px** —
*lower* resolution than our 336. That is published evidence against the resolution hypothesis
(`knee-dft`), recorded there before that arm runs so the result cannot be rationalised afterwards.
A 2025 multi-plane study also found single-plane sagittal (0.976) beating three-plane fusion (0.949)
for ACL, which tempers the extended-slot arm (`knee-t92`): more sequences is not automatically more
signal.

**What none of them address.** Our actual binding constraint is supervision — 58 of 4,407 studies
carry an image-read annotation and the rest are LLM reads of free text in twelve languages. Every
method above assumes clean radiologist labels. That is why architecture ideas from this literature
should be taken as cheap experiments rather than as a path to the frontier.

Sources: [MRNet](https://stanfordmlgroup.github.io/projects/mrnet/) ·
[Medical Slice Transformer](https://www.nature.com/articles/s41598-025-09041-8) ·
[Memory-Guided Transformer](https://sciencedirect.com/science/article/abs/pii/S0031320325000779) ·
[multi-scale knee injury detection](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full)
