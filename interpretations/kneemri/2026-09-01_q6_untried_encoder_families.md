# Q6 (knee-nz34): why only DINO and RAPTOR — enumerating untried encoder families and pricing each

**Premise correction:** the repository is not DINO-only. `kneemri/models/backbone.py` already registers twelve variants across eight families:

| Family | Variants in registry | Mount status |
|--------|---------------------|--------------|
| DINOv2 | small, base, large | Public (`metaresearch/dinov2`) |
| DINOv3 | v3small, v3splus, v3base | Private (`dimiturnt/dinov3-hf`) |
| ConvNeXtV2 | cnxbase | **None** |
| Swin | swinbase | Mirrored (`dimiturnt/swin-base-384`) |
| ViT IN-21k | vitin21k | **None** |
| V-JEPA2 | vjepa2l | Mirrored (`dimiturnt/vjepa2-hf`) |
| RAD-DINO | raddino | Mirrored (`dimiturnt/rad-dino`) |
| RadImageNet | radnet121 | **None** |

The pool is DINO-*dominated* at ~84% of scored members, not DINO-only.

## What each family actually reads on our holdout

Best single run per variant on the 909-study fold-0 holdout, using whatever recipe produced the number:

| Variant | Best run | Holdout AUC | Gap to DINOv2-base `base_dense_s2026` |
|---------|----------|------------:|--------------------------------------:|
| small (DINOv2-S) | `ema_dense_s2026` | 0.8588 | +0.0076 |
| base (DINOv2-B) | `base_clean_s2026` | 0.8570 | +0.0058 |
| base (DINOv2-B) | `base_dense_s2026` | 0.8512 | 0.0000 |
| v3base (DINOv3-B) | `v3base_dense_s2026` | 0.8455 | −0.0057 |
| vjepa2l | `vjepa_member_s2026` | 0.8343 | −0.0169 |
| swinbase | `raptor_feat_fit_s2026` | 0.8334 | −0.0178 |
| swinbase | `swin_supervised_s2026` | 0.8331 | −0.0181 |
| cnxbase | `cnx_own_recipe_s2026` | 0.8186 | −0.0326 |
| raddino | `raddino_normfix_s2026` | 0.8155 | −0.0357 |
| vitin21k | `vit_in21k_s2026` | 0.7847 | −0.0665 |
| radnet121 | `radnet_family_s2026` | **0.6681 @ epoch 9 (in flight)** | −0.1831 |

All numbers come from `D:/data/volumetric/mri/processed/rsna_knee_abnormality/<run>.metrics.jsonl`, best `auc` across epochs. The RadImageNet row is an **incomplete run** — it was still on the card when this was written. Quoting it as a family score would be the failure `convergence_check` exists to stop.

## Pricing each unattempted family

| Family | Why it was added | Cost to try | Kaggle feasibility | Verdict from runs |
|--------|-----------------|-------------|-------------------|-------------------|
| **ConvNeXtV2 (cnxbase)** | First non-transformer; different inductive bias (local conv + hierarchy) vs. global attention | ~1–2 card-hours at 352 px; no mount work yet | Needs dataset upload (`facebook/convnextv2-base-22k-224`, Apache-2.0) | **Recipe-bound**, not quality-bound: best run (`cnx_own_recipe_s2026`, 0.8186) beats the naive dense recipe (`cnx_dense_s2026`, 0.7927) by +0.026. The trunk can be pushed closer but is unlikely to catch DINOv2 base. |
| **Swin (swinbase)** | Supervised IN-22k objective + shifted-window architecture; shippable mount | ~1–2 card-hours at 384 px (pinned read size) | Ready (`dimiturnt/swin-base-384`) | **Quality-bound at ~0.833**: two independent recipes (`raptor_feat_fit_s2026`, `swin_supervised_s2026`) land within 0.0003, suggesting the trunk itself tops out below DINOv2 small. |
| **ViT IN-21k (vitin21k)** | Objective-axis control: same architecture as DINOv3-B, different pretraining objective | ~1–2 card-hours at 224 px | Needs dataset upload (`google/vit-base-patch16-224-in21k`, Apache-2.0) | **Quality-bound**: 0.7847 is far below its architectural twin DINOv3-B (0.8455). Supervised IN-21k is worse for this task than DINOv3 self-distillation. |
| **V-JEPA2 (vjepa2l)** | First tubelet trunk; temporal/video pretraining applied to spatial MRI depth | ~2–3 card-hours; 326M params, fpc64 tubelet reduces memory | Ready (`dimiturnt/vjepa2-hf`) | **Quality-bound but close**: 0.8343 is the strongest non-DINO trunk, but still −0.017 below DINOv2 base. The video pretraining does not transfer to knee MRI depth enough to win a seat on solo strength. |
| **RAD-DINO (raddino)** | First medical-image pretraining; exact ViT-B/14 control vs. `base_dense_s2026` | ~1–2 card-hours | Ready (`dimiturnt/rad-dino`) | **Recipe-bound + normalization-bound**: `raddino_dense_s2026` read 0.8141 because it was fed ImageNet normalization; `raddino_normfix_s2026` with the trunk's own (0.5307, 0.2583) normalization rose to 0.8155. Even fixed, it is −0.036 below the exact DINOv2-B control, so radiology chest-init alone does not move the needle for knee MRI. |
| **RadImageNet (radnet121)** | Medical-image + supervised + different architecture (DenseNet-121); the empty cell in the pretraining matrix | ~1–2 card-hours + 2–4 h PyTorch port/Keras conversion; 7M params | Needs dataset upload + conversion (`Lab-Rasool/RadImageNet`, MIT) | **Quality-bound and heading to empty**: `radnet_family_s2026` was 0.6681 at epoch 9 with two epochs left; its pre-registered middle branch (0.75–0.8186) is out of reach. The 7M-param DenseNet-121 is too small for the task. Published knee MRI numbers from RadImageNet ([PMC9530758](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)) report AUC 0.85 ± 0.09 on ACL/meniscus, but that is in a different setup and does not rescue the run here. |

## Why the non-DINO lane scores worse

1. **Natural-image pretraining dominates.** The best trunks (DINOv2 small/base, DINOv3 base) are all pretrained on large natural-image corpora. The medical-init trunks (RAD-DINO, RadImageNet) and supervised natural trunks (Swin IN-22k, ViT IN-21k) do not generalize better to this knee MRI task.
2. **Self-supervised distillation beats supervised pretraining for this head.** DINOv3-B (self-distilled on LVD-1689M) outperforms ViT-B/16 IN-21k by +0.0608, and DINOv2-B outperforms Swin-B IN-22k by +0.0181.
3. **Domain match is weaker than expected.** RAD-DINO is fine-tuned from DINOv2-B on 882k chest radiographs; with correct normalization it still trails DINOv2-B by 0.036. RadImageNet, trained on 1.35M radiology images, is heading below 0.75 on this task. Knee MRI is far enough from the source domains that medical pretraining does not transfer cleanly.
4. **Architecture diversity has not bought ensemble diversity.** ConvNeXtV2 and V-JEPA2 make different *mistakes* in principle, but their solo strength is too low to earn a seat: the board has repeatedly shown that ensemble seats require both distinctness and solo strength above ~0.852.

## Conclusion

The non-DINO lane is **quality-bound, not recipe-bound, for the trunks that are shippable today** (Swin, V-JEPA2, RAD-DINO): no recipe variation on disk closes the gap to DINOv2 base. ConvNeXtV2 looks recipe-bound in isolation, but its ceiling (~0.819) is still below the seat threshold. The only open bet is whether a *trained* (not frozen) non-DINO trunk could close the gap, but the current evidence says the pretraining objective and corpus matter more than architecture for this task, and the project has already exhausted the obvious pretraining axes.
