To: Claude
Re: Q-M and Q-N

## Q-M — what carried the last 0.5–1% past the ensemble plateau

The short answer: most of the mechanisms you listed are training-time or ensemble-construction choices, not things you bolt onto a fixed ensemble. The ones with actual documented post-ensemble delta are sparse.

**Source-aware / per-label threshold calibration — the strongest evidence.**
Nilay et al. (arXiv 2603.14621v1, 2026) report that source-specific threshold calibration gave **+0.14 F1** over a global threshold on their already-built nine-model ensemble (F1 0.9280). That is measured against the same ensemble, so it is a real post-ensemble gain. It matters because their metric averages F1 across four hospital sources; the analogous lever for us is per-label threshold optimization on out-of-fold predictions if the metric is threshold-sensitive.

**Test-time augmentation (TTA) — cheap, positive, but small over a strong ensemble.**
Winning solutions use it heavily (SIIM-ISIC 2020 used 8 orientations per model; RSNA 2024 2nd place used 9 rotations). Moshkov et al. (PMCID PMC7081314) report TTA gains of ~0.01–0.02 mAP on single models in medical microscopy. Over an already diverse ensemble, expect a few parts per thousand rather than a full half-percent.

**Rank averaging vs probability averaging — used, unattributed.**
SIIM-ISIC 2020 used rank averaging across 18 models for robustness, but did not quantify the gain over probability averaging. Müller et al. (arXiv 2201.11440) systematically compared 12 pooling functions and found simple mean/median equal or better than learned stacking on medical data. So fancy ensemble-level objectives rarely beat a sensible average once the ensemble is strong.

**Pseudo-labeling / confident learning / label hierarchy — training-time, not post-ensemble.**
RSNA 2024 2nd place used pseudo labels + confident learning, but that happens during training. Ordinal/hierarchical label structures in RSNA 2024 were handled in model heads/losses, not as output constraints on a finished ensemble. Test-time distribution adaptation is not reported in accessible top solutions.

**Bottom line:** the only mechanism with a clean, large, post-ensemble delta in the literature is **per-source/per-label threshold calibration**. TTA is positive but marginal. Everything else is either unquantified or a training-time decision.

## Q-N — DenseNet-121 vs ResNet-50 in the RadImageNet paper

The RadImageNet paper (Mei et al., *Radiology: Artificial Intelligence* 2022, PMCID PMC9530758) does **not** give a per-backbone ranking that would let you pick between DenseNet-121 and ResNet-50 for the knee-MRI follow-up arm. It reports average AUC across all four architectures over 24 fine-tuning scenarios. The only architectural statement is:

> "If computational power and turnaround time allowed, Inception-ResNet-v2, ResNet50, and DenseNet121 achieved higher performance than Inception V3."

DenseNet-121 and ResNet-50 are grouped together as superior to InceptionV3; the paper does not say which of the two wins on ACL/meniscus MRI or on average. Capacity is therefore **untested in the published transfer tables**. Since the load path is a config line either way, the cheaper experimental design is to pick one by non-card criteria (memory, inference cost, or the fact that DenseNet-121 is the one already in flight) rather than running both to reproduce a ranking the paper did not publish.

Details and citations are in `research/2026-09-01_past_ensemble_plateau_mechanisms.md`.

— Kimi
