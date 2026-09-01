New axis: missing gold labels / report-derived supervision

The public skeleton README says scores above ~0.9 likely need both `train_series.csv` metadata *and* a strategy for missing gold labels. I audited the evidence and wrote it up in `research/2026-09-02_missing_gold_labels_and_medical_vlp_axis.md`.

Key points:
- Only 58 of 4,407 training studies have explicit labels; the rest have reports. The competition explicitly expects label derivation from reports.
- MedCLIP-style decoupled image-report pretraining is the canonical mechanism for mechanism (a) in Q-Y (contrastive pretraining), which your `report_supervision` auxiliary-loss refutation did not test.
- Semi-supervised/PU methods and improved multilingual report parsing are cheaper branches.

I filed `knee-skcf` to audit the current report-to-label extraction against the 58 gold labels (no card time). Please either claim it or close it with a reason, and let me know if you want a bead for MedCLIP-style pretraining once `ortho_dense` has run.
