# A-DS — 3D-native encoders: the family is shippable, the mechanism is not supported

**Asked:** Q-DS, 2026-09-04. Every member this project owns is a 2D encoder run per slice with a pooling
operator over the stack. Is there a natively volumetric encoder that would (a) ship and (b) decorrelate
from a 2D slice-pooled reader? Answered by Kimi the same hour, no card spent.

## The verdict

**The axis closes.** Not on licence and not on availability — a shippable candidate exists — but on
published evidence that the mechanism does not deliver, plus a total absence of the decorrelation
evidence that would have justified the arm as a diversity play.

## What is shippable

| Candidate | Class | Pretraining modality | Licence | Verdict |
|---|---|---|---|---|
| **MedicalNet 3D-ResNet** (`TencentMedicalNet/MedicalNet-Resnet{10..200}`) | 3D ResNet, single-channel volume, classification head | 23 medical datasets incl. brain MRI; **no knee/MSK** | **MIT**, GitHub `LICENSE` and HF card agree, no `config.json` to conflict | The only shippable 3D-native *classification* trunk |
| **SwinUNETR SSL** (`Project-MONAI/MONAI-extra-test-data`) | 3D Swin encoder + U-Net decoder, ~50M encoder | ~50k 3D CT/MR incl. BRATS, breast MR, prostate MR | **Apache 2.0** | Shippable, but segmentation-origin — drop the decoder, add a head |
| SegVol (`BAAI/SegVol`) | Interactive 3D segmentation | 90k CT unlabeled + 6k labeled, **CT only** | MIT | Not classification, wrong modality |
| M3D-CLIP / M3D-LaMed | 3D medical VLM | mixed 3D medical | code MIT, **weights gated (401)** | Instrument at best |
| VISTA3D | 3D interactive segmentation | CT-focused | **gated (401)** | Out |
| Models Genesis | 3D ResNet/VNet pretraining | multiple medical | **non-commercial** | Not shippable |
| VideoMAE / V-JEPA / InternVideo | natural-video transformers | Kinetics-400 etc. | VideoMAE **CC BY-NC 4.0** | Not medical, not shippable |
| CT-CLIP / CT-FM / Merlin | CT foundation models | **CT only** | CT-FM gated | Wrong modality |
| OmniRad | — | — | `cc-by-nc-4.0` conflict (Q-DR) | Already rejected |

MedicalNet geometry: `input_D=56, input_H=448, input_W=448` default, single-channel volume, standard
PyTorch 3D-ResNet keys (`conv1`, `bn1`, `layer1..4`, `fc`). SwinUNETR: `(96,96,96)` patches, MONAI
`swinViT.*` keys after mapping.

## Why the axis closes anyway

Three published readings, in increasing order of how much they bind:

1. **Pedoia et al. 2018** (*JMRI*, PMC6521715) — a custom 3D CNN on 1,478 knee MRI studies detects and
   stages meniscus and patellofemoral cartilage lesions (meniscus 89.8% / 82.0% sens/spec). A real
   knee-MRI 3D result, but not MedicalNet and not our labels. It establishes the family works at all,
   nothing more.
2. **Namiri et al. 2020** (arXiv:2003.09089) — 2D CNN vs 3D CNN for ACL severity staging on knee MRI,
   the closest published head-to-head to our task. **2D: 92% accuracy, kappa 0.83. 3D: 89%, kappa 0.83.
   P = 0.27.** 2D was not worse, on the same data, on knee MRI.
3. **AnyMC3D / Liu et al. 2025** (arXiv:2512.12887) — a twelve-task 3D medical classification benchmark.
   Properly adapted 2D foundation models reach **0.894 average AUC** against **MedicalNet 0.783** and
   **VoCo 0.793**. Their own conclusion: *"2D-based methods surpass 3D architectures for 3D
   classification."*

No published result shows a 3D foundation model beating a strong 2D slice-pooled reader on knee MRI
abnormality classification.

## The decorrelation question, which is the one that mattered

Searched for error correlation, inter-model kappa, Q-statistic, or cross-architecture ensemble gain
between a volumetric encoder and a 2D-per-slice encoder on the same task. **Nothing.** Namiri reports
per-model kappa against ground truth, not between models. AnyMC3D ensembles ten models but never breaks
out a 2D-vs-3D pair or reports a diversity statistic. Hybrid 2D+3D papers exist (brain tumour grading,
lung cancer) but measure no diversity statistic and are not knee MRI.

So the mechanism the arm would have been bought for — a 3D reader erring where a 2D slice-pooled reader
does not — is **unestablished**, not merely unmeasured here.

## What this does NOT rule out

The family is closed for now; the asset is not dead. Re-open it by answering the argument, not by
re-quoting a number: a measured error correlation between a cheap MedicalNet-ResNet10 member and our
best 2D member would be the thing that does it, and it is a smoke-sized job rather than a full arm.

## The lever this opened, and why the ledger closed it the same hour

AnyMC3D beats 3D foundation models **with a 2D backbone**, so its win lives in the ADAPTATION — how
per-slice features become one study-level prediction — not in the trunk. That pointed at slice fusion,
and the ledger had already answered it. Checked before arming anything:

- `attend_true` (`experiments/README.md:302`) — a 64-wide gate the optimiser actually reaches,
  plumbing verified (`slices.gate.3.weight` absmax 8.60e-02 against `attend_dense`'s exact 0.000e+00):
  **0.8476**, +0.0003 vs the mean.
- `depth_true` (line 303) — `DepthAttention`, the only aggregator that sees slice ORDER: **0.8461**,
  −0.0015 vs `attend_true`, second-worst of four.
- Elementwise max 0.8457, unweighted mean 0.8473.

**Four operators on one recipe span 0.0019** — under a third of the 0.0062 holdout floor. And the
stage is saturated from three directions at once: more inputs (`knee-ezz`, 20 cached slices vs 12:
−0.0001), more slots (`extended_slots`, 83,297 unseen slices: +0.0002), every combining rule (0.0019).
The limit is upstream, in what one slice's feature vector contains.

So the fusion arm does not get the card. The one exclusion left was that all four of our operators fuse
POOLED PER-SLICE VECTORS, and a trunk attending over the joint patch-token set across slices would be a
different mechanism. **A-DT answered it the same day and it is not that**
(`2026-09-04_qdt-slice-fusion-mechanism.md`): AnyMC3D encodes each slice and takes the **CLS token** as
the slice embedding, then attends over the stack of those with one learned task query, permutation-
invariant, no positional encoding. Patch tokens go to an auxiliary segmentation branch and never enter
the classification fusion. Their operator is our `attend_true` with a single query instead of a 64-wide
gate — and their own trunk-held ablation reads +0.004 over average pooling on one CT task, against our
+0.0003 over the mean on ours.

**The axis is closed on published and local evidence together.** Both the mechanism and the size of the
effect agree; the remaining difference is a knob, and the ledger has already measured that knob here.
