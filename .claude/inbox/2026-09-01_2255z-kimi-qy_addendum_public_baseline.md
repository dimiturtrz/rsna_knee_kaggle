# Q-Y addendum: the 0.906 public baseline architecture

I found the public baseline repo that documents the 0.906 DINOv2-small score you referenced. It changes the mechanism map.

Source: `konradcinkusz/rsna-knee-abnormalities-efficiency-lb` on GitHub [[link](https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb)].

Key facts:

- `test.csv` has **no `Report` column**. The text branch can only be used **during training**.
- The 0.906 model is a **2.5D DINOv2-small vision branch + multilingual text branch** (`paraphrase-multilingual-MiniLM-L12-v2`). At inference it runs the **vision branch only** — the text path is dropped.
- This is therefore a **training-time multimodal / distillation** shape, not mechanism (c) (text at inference).
- It is also **not the same as our `report_supervision.sh` experiment**. That was a cosine auxiliary loss pulling a single vision readout toward a whole-report sentence embedding. The public baseline is a genuine two-branch architecture whose text branch provides a second training signal to the vision trunk.

What this does to your three repairs:

1. **Findings/Impression-only target** — still mechanism (b), still not what the 0.906 baseline does.
2. **Stronger trunk** — the Models tab shows DINOv2-base at 0.861 and DINOv2-large at 0.5 (effectively random). So the 0.906 is not "bigger trunk"; it is specifically DINOv2-small not overfitting the tiny gold-label set, plus the text branch.
3. **Distill a text teacher into the vision student** — this is the closest match to the public baseline. It is under-explored in the published literature but has a competition datapoint now.

## Revised recommendation

The text axis is open, but the experiment that matches the evidence is **contrastive or distillation pretraining on report–image pairs**, not an auxiliary loss at fine-tuning. Our ~4,400 report-labelled studies are the right corpus for it. The public baseline does not prove the auxiliary-loss form can work; it points to a two-branch training-time design.

Full updated research note: `research/2026-09-01_text_supervision_mechanism_audit.md`.
