"""Two members through the corrected readout: `experiments.arms.foreign_drop_verify`.

WHY A SEPARATE ENTRYPOINT. `foreign_readout` sizes its read window against the depth the slab SERVES
(`HoldoutSlab.depth`) rather than the depth its source corpus holds. That was a one-line defect with two
faces, and this exists to show both are gone in one run: a 224 px member (12 slices) whose foreign side
used to CRASH on `view[:, :, 16:28]` of a 12-deep stack, and a 336 px member (20 slices) whose foreign
side used to SCORE SILENTLY on `view[:, :, 12:32]` — eight slices of twenty, with a plausible number
logged for it. The second is the one that reached a reading, so a fix is only demonstrated by a pair.

PRE-REGISTERED: both members must produce a reading, `slot_cost` must stay inside +/-0.005 on both, and
`stackfit`'s decode_cost must MOVE from the -0.0834 taken on the truncated stack. A stackfit that reads
-0.0834 again would mean the window never changed and the fix did not land.
"""

import sys

from kneemri.evaluation import foreign_readout
from kneemri.inference.holdout_predictions import HoldoutPredictions

MEMBERS = ("native224_s2026.pt", "stackfit_s2026.pt")


def main() -> None:
    """Run the two members and print the window each side actually reads."""
    predict = HoldoutPredictions.predict

    def traced(
        self: HoldoutPredictions, model: object, slab: object, window: object, precision: object = None
    ) -> object:
        print(  # noqa: T201
            f"WINDOW slab={type(slab).__name__} depth={slab.depth} "  # type: ignore[attr-defined]
            f"starts={window.starts} width={window.width} size={window.size}",  # type: ignore[attr-defined]
            flush=True,
        )
        return (
            predict(self, model, slab, window) if precision is None else predict(self, model, slab, window, precision)
        )

    HoldoutPredictions.predict = traced  # type: ignore[method-assign]
    models = "D:/models/rsna_knee_abnormality"
    processed = "D:/data/volumetric/mri/processed/rsna_knee_abnormality"
    sys.argv = [
        "foreign_readout",
        "--checkpoints",
        *[f"{models}/{name}" for name in MEMBERS],
        "--labels",
        f"{processed}/labels/report_labels_v2.csv",
        "--foreign",
        "D:/kaggle/input/raptor-corpus",
        "--device",
        "cuda",
    ]
    foreign_readout.main()


if __name__ == "__main__":
    main()
