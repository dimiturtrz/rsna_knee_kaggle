# Q-Y: text-supervision gains live in pretraining, not in auxiliary loss or inference

You asked whether the published gains from report supervision sit in (a) contrastive pretraining, (b) auxiliary alignment loss during supervised fine-tuning, or (c) a text branch at inference. The answer is almost entirely **(a)**.

## What the literature shows

- **ConVIRT, CheXzero, MedCLIP, GLoRIA, BioViL/BioViL-T** all train vision + text encoders jointly on large image–report corpora. The downstream gains come from the **pretrained vision encoder** (fine-tuned or frozen) or from zero-shot matching. The text encoder is either discarded at fine-tuning time or reduced to class-name prompts at inference.
- **MedCLIP** is the cleanest numbers: on four chest-X-ray tasks, fine-tuning its pretrained encoder beats an ImageNet init by large margins (e.g., CheXpert-5x200 ACC 0.596 vs 0.320; MIMIC 0.565 vs 0.283). But this is still **pretraining**, not an auxiliary loss added during supervised fine-tuning.
- **Mechanism (b)** has no published medical gain I could find. The closest general-domain paper is *Improved Visual Fine-tuning with Natural Language Supervision* (TeS+CLIP, ICCV 2023), which reports +4.4% avg accuracy on fine-grained datasets using text supervision during visual fine-tuning — but it uses class-name prompts, not free-text reports, and it is not medical.
- **Mechanism (c)** (text branch at inference) is common in report generation / VQA, but not reported as a diagnosis-classification accuracy lever.

Full research note: `research/2026-09-01_text_supervision_mechanism_audit.md`

## What this means for your three repairs

1. **Findings/Impression-only or per-label prototypes.** This is still mechanism (b). It might rescue some of the −0.0063 by removing noisy tokens, but the literature gives no reason to expect it to turn positive against a strong label-supervised model.
2. **Stronger trunk control.** Changing the baseline does not change the mechanism. A stronger trunk may shrink the relative damage, but it does not create a gain where the literature does not report one.
3. **Distill a text teacher into the vision student.** This is the most interesting because the teacher itself would be a mechanism-(a) model. General-domain CLIP distillation works, but medical diagnosis distillation from reports is essentially unreported.

## Recommendation

If you want to keep the text axis open, the only literature-backed move is **contrastive pretraining on the project's own ~4,400 report-labelled studies**: continue the DINOv3-L (or OrthoFoundation-L) trunk with an image–report contrastive objective, then fine-tune on the 12-label task. That is mechanism (a), and it has direct precedent.

Mechanism (b) as you tested it is consistent with the literature — which mostly does not claim it works — so I would **decline the auxiliary-loss form** unless you want a cheap Findings-span ablation to close the door completely.

Bead tracking this: `knee-iem6`.
