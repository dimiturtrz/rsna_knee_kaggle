# Frozen-BatchNorm drift in transfer learning (Q-DF)

Observation (Claude): a frozen RadImageNet trunk had `num_batches_tracked = 226711` and its BatchNorm statistics were effectively replaced by target-domain statistics, voiding the claimed transfer result.

## Why it happens

In PyTorch, `BatchNorm` layers update their `running_mean`, `running_var`, and `num_batches_tracked` buffers during forward passes in `train()` mode. `requires_grad=False` only freezes the affine `weight` and `bias`; it does **not** freeze the running statistics. With momentum 0.1, source-domain statistics are overwritten almost immediately.

## Standard fixes

1. **Backbone `.eval()` during frozen phase** — simplest; also disables dropout.
2. **Selective BN-eval** — keep model in `.train()` but force every `_BatchNorm` module to `.eval()`. This is the `norm_eval=True` pattern in MMDetection/MMSegmentation.
3. **AdaBN** — deliberately re-estimate BN statistics on the target domain instead of preserving source stats. More principled when domain shift is large.

## Implication for our pool

The RadImageNet transfer result (0.6716) is not a clean architecture-transfer measurement; it reflects RadImageNet architecture with knee-MRI BatchNorm statistics. It should be re-run with BN buffers frozen before being used as evidence about RadImageNet features.

## Sources

- PyTorch Forums on BN eval during fine-tuning: https://discuss.pytorch.org/t/should-i-use-model-eval-when-i-freeze-batchnorm-layers-to-finetune/39495
- MMSegmentation `norm_eval=True` implementation: https://github.com/open-mmlab/mmsegmentation/issues/2447
- Li et al., *Revisiting Batch Normalization For Practical Domain Adaptation* (AdaBN): https://ar5iv.labs.arxiv.org/html/1603.04779
