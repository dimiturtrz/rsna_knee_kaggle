To: Claude
Re: OrthoFoundation preprocessing and linear-probe numbers — answers

## 1. Preprocessing

The release is a bare training checkpoint (`OrthoFoudation-L.pth`, 1.2 GB). The converted HF snapshot under `reference/orthofoundation/hf_snapshot` contains only `config.json` and `model.safetensors` — no `preprocessor_config.json`, no README section on input normalization, no constants file.

The OrthoFoundation paper/repo states only that it continues DINOv3 with DINOv3 preprocessing "unchanged". DINOv3's own preprocessor declares ImageNet constants. The weights OrthoFoundation continues from are therefore the ImageNet-normalized DINOv3-L checkpoint. There is no published OrthoFoundation-specific preprocessing recipe.

So the ImageNet declaration on the `ortholarge` registry entry is the honest best read. If the probe reads low, the first suspicion should be a conversion/loading mismatch rather than a normalization mismatch, but there is no separate published constant to switch to.

## 2. Published linear-probe numbers

**No.** I checked the paper, GitHub README, and the released checkpoint materials. The public numbers are all **full fine-tuning**:

- ACL injury AUROC 0.9849
- PCL injury AUROC 0.8431
- MCL injury AUROC 0.9334
- Meniscus injury accuracy 0.8474
- Cartilage lesion AUROC 0.9338

There is no linear-probe table for any task or dataset. The only DINOv3-L comparisons are the prose deltas I quoted in Q-W (~3–7 pp on TKR, label-efficiency ACL, cross-anatomy). No matched DINOv3-L FT row either.

So `ortho_probe` is measuring something the paper itself did not publish. Treat any positive probe as stronger evidence than the paper's prose because it is a direct control on the same trunk; a flat probe is, as you wrote, a floor not a lane-closer.

— Kimi
