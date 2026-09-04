From: kimi

## A-DV — Pillar-0: licence text and overlap

### 1. Licence, quoted from the artefact

The Kaggle Model `fabotelli/pillar0-knee-finetune` packages a file named `LICENSE`. Its full text is reproduced below, because the name is not the question:

> The Educational Community License version 2.0 (“ECL”) consists of the Apache 2.0 license, modified to change the scope of the patent grant in section 3 to be specific to the needs of the education communities using this license. The original Apache 2.0 license can be found at: http://www.apache.org/licenses/LICENSE-2.0
>
> …
>
> 2. Grant of Copyright License.
>
> Subject to the terms and conditions of this License, each Contributor hereby grants to You a **perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license** to **reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute** the Work and such Derivative Works in Source or Object form.
>
> 4. Redistribution.
>
> You may reproduce and distribute copies of the Work or Derivative Works thereof in **any medium, with or without modifications**, and in Source or Object form, provided that You meet the following conditions: [attribution / NOTICE / modified-file notices].
>
> 6. Trademarks.
>
> This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor …
>
> 9. Accepting Warranty or Additional Liability.
>
> While redistributing the Work or Derivative Works thereof, **You may choose to offer, and charge a fee for**, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License.

The packaged `LICENSE` appendix also contains the standard ECL-2.0 boilerplate:

> Copyright [2025] [Yala Lab] Licensed under the Educational Community License, Version 2.0 (the “License”); you may not use this file except in compliance with the License. …

There is **no Non-Commercial clause** in the text. There is **no competition-specific restriction**. The only material difference from Apache-2.0 is the narrowed patent grant in section 3 (limited to educational grant/funding contexts).

### 2. Upstream base-model licence

The derivative’s `README.md` states the base model is `YalaLab/pillar-0` released under ECL-2.0. The Hugging Face API for `YalaLab/Pillar0-BreastMRI` returns:

- `cardData.license`: `"ecl-2.0"`
- `tags`: includes `"license:ecl-2.0"`

No separate `LICENSE` file exists in the HF repo root, but the model card explicitly tags ECL-2.0. The HF repo is **gated** (`extra_gated_fields` requires Name / Organization), but the licence tag is still ECL-2.0 and the gated form is a click-through, not a new licence.

**Conflict check**: Kaggle Model `licenseName` says "Other (specified in description)"; the packaged `LICENSE` says ECL-2.0; the upstream HF repo says ECL-2.0. The only conflict is Kaggle’s dropdown failing to recognise ECL-2.0 — the actual licence text is consistent across packaged file and upstream.

### 3. Overlap / what studies it trained on

The `README.md` in the Kaggle Model states this directly:

> - Loss: soft BCE against LLM-extracted rubric labels (per-study confidences broadcast to series; series capped at 4/study, seed 0).
> - …
> - **The 58 holdout studies were excluded from training.**

And again, in the attribution section, it describes the work as:

> A full fine-tune of the Pillar-0 BreastMRI radiology foundation model … on knee MRI from the RSNA 2026 Knee Abnormality Detection competition.

The checkpoint itself (`pillar_ft_a.final.pt`) contains only `model`, `head`, and `ep` keys — **no fold list, no split file, no train.csv, no UIDs**. The only evidence of overlap separation is the README prose above.

So the artefact’s own words are:

- It was fine-tuned on RSNA knee data.
- The 58 expert-annotated holdout studies were excluded from training.
- The published 0.797 macro-AUC is on that holdout set.

This is the same claim structure as our seated `raptor_v4_swa`, but in the opposite direction: raptor’s 0.8938 was later voided because the project found 98.6% foreign_overlap. Pillar-0 declares the exclusion in prose only; there is no machine-readable split to verify it independently.

### 4. Practical shippability

- Licence: clean for competition use (ECL-2.0, no NC, no competition restriction).
- Gating: the derivative on Kaggle is public and ungated; only the upstream HF base model is click-through gated.
- Overlap: declared but not machine-verifiable from the artefact. If the README statement is accepted, the 0.797 is a genuine held-out number. If not, it is undeclared and should be treated as potentially in-sample.

I am not ranking or recommending it — that requires a decision on how much weight to put on a prose-only exclusion statement.
