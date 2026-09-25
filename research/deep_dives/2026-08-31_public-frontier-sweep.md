# The public frontier, swept again — what it confirms, and the one cell it says we have never read

2026-08-31. Thirty newest public notebooks on `rsna-knee-abnormality-detection` (`kaggle kernels list
--competition ... --sort-by dateRun`), four pulled and read in full. The forum API answers 403 for
discussion topics, so this sweep is notebooks only; `research/deep_dives/2026-08-25_competitor-single-model-scores.md`
remains the forum-side record.

The point of the sweep was to find an axis we have not tried. Most of what it surfaced is a
CONFIRMATION of something already in our ledger, which is worth as much — an independent stack reaching
our conclusion is stronger evidence than our own repeat of it would be.

## What it CONFIRMS, from a stack that shares nothing with ours

**Ensembling is worth ~+0.001 on this competition's board.** `hyakumanben2025/rsna-knee-dinosaur-v5-repro`
(the raptor lineage) states it as its own reason for shipping a single arm:

> *"on the live leaderboard CoAtNet alone scored 0.914 while every blend scored 0.914-0.915, so ensembling
> is worth ~+0.001 there — the ~+0.010 it showed on the old 45-study gold set was gold-set noise. One arm
> is also 1/3 the kernel runtime."*

Their panel is three architectures — CoAtNet 384, Swin-base 384, EffNetV2-L 480 — a genuinely
cross-family blend, and it still buys +0.001. Ours is six DINOv2-small recipes and buys the same. Two
unrelated stacks, one number. This is the same finding as `experiments/README.md`'s
*"every operation on the champion's seating is now measured, and all of them are flat"* (three offline
zeros, `holdswap_v1` 0.944 dead flat, `raptorseat`/`vjepaseat`/`fleettwo` 0.942-0.943), and as
knee-va22 and knee-xuj5. **It should retire the suspicion that our seating flatness is an artefact of
our monoculture.** It is not: a three-family blend is flat too.

**Their corpus is our corpus, and they have already spent the volume lever.** They rebuilt from 3,155 to
4,349/4,407 studies (+37.8%) and measured +0.0131 on their 58-study gate, better in 92.7% of 2,000
bootstraps. We are at 4,407. The competition's labelled corpus is exhausted on both sides, which is why
`knee-ombx` (MRNet, external, image-read ACL **and** meniscus labels) is the only volume lever left and
why it is argued on KIND rather than count.

**Their gains landed on the thin findings.** *"Lateral Meniscus +0.071, Fracture +0.057, Lateral OA
+0.048, Medial Meniscus +0.035, ACL +0.028."* Same set our own `merged_corpus` arm moved (Lateral
Meniscus +0.0196\*, ACL +0.0161) and the same set `2026-08-25_the-focal-deficit.md` calls the deficit
findings. Three independent mechanisms now point at the focal structures.

## What it surfaced that we had already covered — checked, not assumed

| Public idea | Where we already stand |
|---|---|
| **CoAtNet** (`coatnet_rmlp_2_rw_384`), the frontier's best single | It IS `raptor_v4_swa`. We seat it (`RaptorSeat`, weight 1/3) and distil from it (`raptor_teacher.npz`). knee-nj4l: it scored 0.923 solo on the same day our own best single scored 0.923. Training our own copy would buy an ordering we already hold |
| **Multilingual weak supervision** (`eriktavares`, a hand-built 8-language lexicon) | `kneemri/embeddings/report_embeddings.py` embeds all 4,407 reports with `paraphrase-multilingual-MiniLM-L12-v2`; `kneemri/evaluation/unsure_cost.py` reports English and non-English pairs separately. Covered, and by a stronger instrument than a regex lexicon |
| **Multi-series / per-plane fusion** (`qunnguynm`, 3D-MIL over sagittal/coronal/axial slots) | `kneemri/evaluation/raptor_feed.py` fills six plane/weighting slots; `SlotHead` attends over them per finding; `DepthAttention`/`DepthWindow` never mix across a slot boundary. Covered, and the order-aware variant was measured and refuted (`depth_true`, −0.0015) |
| **2.5D three-central-slices ResNet18 baselines** | Below every member on our roster |

## The one cell nobody here has ever occupied: slice COVERAGE

The frontier fills its five slots with **64 slices** per study (18+14+12+8+12). Across our **137 arm
scripts** the slice count is 20 (95 arms), 12 (43), 16 (1), and **nothing above 20**. A sagittal knee
series holds 30-40 slices, so every member this project owns has read a third to a half of the volume
with a sampler choosing which half.

The slice-count readings we hold do not answer this, because all of them are 12 → 20: +0.0080
(`band_span_hold` → `band_dense_hold`), +0.0013 (`band_span` → `band_dense`), −0.0001 under max pooling
(knee-ezz), −0.0028 on the merged corpus. That is a change of WHICH SUBSET at fixed partial coverage.
Reading 36 of a 30-40 slice stack is the first read that cannot miss a slice — a different mechanism, and
the only one that can answer whether a thin finding is falling between our samples.

It was never affordable: VRAM is a function of slice count alone, and at 336 px twenty slices already
reserve 26.92 GB of 32. At 224 px — the standing exploration resolution since knee-s9cr, where
`native224` cost −0.0023 against its 336 control — the probe build measures **10.84 MB/study** and 36
slices land near 24 GB of VRAM. `band224.sh` was written at 20 slices and never ran, so the 224 slice
axis is untested end to end.

Armed as `stack224` (knee-qphd), read PER LABEL rather than on the twelve-label mean, because the
mechanism names which labels it should move and a mean averages a two-label effect away.

## What the sweep did NOT find

No public notebook above 0.926 LB publishes its method, and the 0.914-0.926 band is all
architecture-panel-plus-blend work we have already priced. The gap between the public notebooks and the
board's top (0.952) is unexplained by anything published, which is the same state
`competition-is-live-no-writeups` recorded.
