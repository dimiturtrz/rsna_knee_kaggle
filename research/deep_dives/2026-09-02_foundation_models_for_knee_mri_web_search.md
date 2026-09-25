# Foundation-model web sweep for knee MRI (2026-09-02)

Quick web search to supplement the existing 2026-09-02 literature notes. No code changes.

## Models / papers found

### Decipher-MR — a vision-language foundation model for 3D MRI (Nature 2026)

- **Reference:** Z. Yang et al., *Decipher-MR: a vision-language foundation model for 3D magnetic resonance imaging of the brain, spine, pelvis and knee*, npj Digital Medicine 2026. [Nature article](https://www.nature.com/articles/s41746-026-02596-4)
- **What it is:** A 3D MRI vision-language foundation model trained on heterogeneous single-contrast volumes across body regions, including the knee.
- **Why it matters:** It is a recent, peer-reviewed foundation model that explicitly covers knee MRI and uses a contrast-agnostic, 3D architecture. The Nature page was blocked by a browser-check wall during this sweep, so full numbers and weight availability were not extracted.
- **Action:** Fetch the PDF or HuggingFace page directly to check whether weights are public and whether knee-specific downstream numbers are reported.

### 3DINO — a generalizable 3D self-supervised framework (CVPR 2026 CT FM Challenge)

- **Reference:** kmin940/3DINO_CVPR26CTFM on GitHub, 2026. [repo](https://github.com/kmin940/3DINO_CVPR26CTFM)
- **What it is:** A 3D self-supervised learning framework and pretrained weights (3DINO-ViT) built for the CVPR 2026 CT Foundation Model challenge.
- **Why it matters:** It is a public 3D self-supervised ViT with weights. The repository focuses on CT, but the architecture and pretraining recipe may transfer to MRI volumes if the slice dimension is treated as the depth axis.
- **Caveat:** No knee or MRI pretraining is claimed; any transfer would require in-domain continuation on fastMRI/MRNet/Rijeka/OAI.
- **Action:** Check the HuggingFace release for weight licence and input-shape constraints.

### Foundation Models for Volumetric Medical Imaging — survey (MDPI 2026)

- **Reference:** T. Ghosh et al., *Foundation Models for Volumetric Medical Imaging*, Electronics 2026. [MDPI article](https://www.mdxpi.com/2079-9292/15/6/1245)
- **What it is:** Survey of foundation models for 3D CT/MRI analysis.
- **Why it matters:** Useful as a source list for 3D architectures and pretrained weights; likely references recent generalist 3D FMs (SegResNet, UNETR, SAM-Med3D, VISTA3D, etc.).
- **Action:** Scan the bibliography for knee/MSK-specific weights or tasks.

### ISMRM 2025 — self-supervised pretraining for 3D knee MRI (OAI)

- **Reference:** ISMRM 2025 session *Applications of Foundation Models*, abstract citing OAI-based self-supervised pretraining. [ISMRM program](https://www.ismrm.org/25/program-files/D-36.htm)
- **What it is:** A conference abstract reporting effectiveness of self-supervised pretraining on 3D knee MRI from the OAI database.
- **Why it matters:** OAI is public domain and large (~4,800 subjects). It is the only external corpus large enough to move SSL meaningfully, per the existing external-data audit.
- **Action:** Find the full abstract or poster to learn the exact pretraining objective and whether weights/code are shared.

## Implications for this project

1. **OrthoDiffusion remains the most actionable 3D candidate** because it is knee-specific, multi-label, and has downloadable weights. Nothing in this sweep displaced it.
2. **Decipher-MR is the strongest generalist contender** to watch: if its weights are public and it reports knee numbers, it could either complement or benchmark OrthoDiffusion.
3. **3DINO and generalist 3D SSL frameworks** are only useful as architecture/pretraining recipes; they would still need in-domain continuation on our external corpora.
4. **OAI continues to be the highest-value unlabelled corpus** for any continuation-based approach.

## Suggested next web/literature actions

- Retrieve the Decipher-MR PDF directly and check for public weights.
- Read the ISMRM 2025 OAI self-supervision abstract for the objective recipe.
- Check 3DINO's HuggingFace licence and input format.
