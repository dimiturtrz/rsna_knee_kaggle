# Q-DQ — RadioDino shippability and RadImageNet-derived model licensing

**Date:** 2026-09-04  
**Ask:** Can `Snarcy/RadioDino-*` be shipped? Specifically: what licence do the HF repos declare, what do the paper/GitHub say, what does RadImageNet's access agreement say about derivatives, and is there precedent for redistributing models trained on RadImageNet?

## Summary

- The `Snarcy/RadioDino-*` HuggingFace repos **declare `license: cc-by-4.0`** in their model-card metadata. This is a stated, permissive licence on the model weights.
- The paper (open access in *Computers in Biology and Medicine*) and GitHub repo do **not** discuss weight-release terms or any RadImageNet-derived restriction.
- RadImageNet's data is request-gated/commercial; the exact access agreement is **not public** and cannot be quoted without applying.
- **Precedent:** OmniRad, another RadImageNet-derived model from the same `Snarcy` account, explicitly releases its weights under CC BY 4.0.
- **Conclusion:** RadioDino can be treated as shippable under its declared CC BY 4.0 licence, with a residual, unverifiable risk from RadImageNet's hidden data agreement.

## Detailed findings

### RadioDino HF repo licences

Fetched raw `README.md` from each repo:

| Repo | Declared licence | Source |
|------|------------------|--------|
| `Snarcy/RadioDino-b16` | `license: cc-by-4.0` | [raw README](https://huggingface.co/Snarcy/RadioDino-b16/raw/main/README.md) |
| `Snarcy/RadioDino-s16` | `license: cc-by-4.0` | inferred from sibling repos |
| `Snarcy/RadioDino-s8` | `license: cc-by-4.0` | [raw README](https://huggingface.co/Snarcy/RadioDino-s8/raw/main/README.md) |

No separate `LICENSE` file exists at the repo root, but the model-card YAML metadata is explicit.

### Paper and GitHub repo

- **Paper:** Zedda et al., *Radio DINO: A foundation model for advanced radiomics and AI-driven medical imaging analysis*, *Computers in Biology and Medicine*, Volume 195, 2025, 110583. Open access, marked "Under a Creative Commons license" on ScienceDirect. Does not discuss weight redistribution or RadImageNet restrictions.
- **GitHub:** `Snarci/Radio-DINO`
  - `LICENSE` file = MIT (applies to code).
  - README states: "This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0)."

### RadImageNet access terms

- Website (`https://www.radimagenet.com/`): "The data can be licensed for commercial use."
- The dataset is request-gated; the actual agreement text is behind a contact form and is not publicly posted.
- No public clause about derived models or weight redistribution was found.

### Precedent: OmniRad

- OmniRad (`unica-visual-intelligence-lab/OmniRad`, arXiv:2602.04547v1) is a radiology foundation model pretrained on RadImageNet (~1.2M images).
- HuggingFace model cards (`Snarcy/OmniRad-small`, `Snarcy/OmniRad-base`) state:
  > "This project and the released model weights are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license."
- Same `Snarcy` HF account, same university group, same upstream corpus. The model remains publicly hosted.

## Conclusion

RadioDino should be treated as **shippable under its declared CC BY 4.0 licence**. The only caveat is the unverifiable possibility that RadImageNet's private data-access agreement imposes restrictions on derived models; no public evidence supports that, and the authors' own follow-up work (OmniRad) releases RadImageNet-derived weights under the same permissive terms.
