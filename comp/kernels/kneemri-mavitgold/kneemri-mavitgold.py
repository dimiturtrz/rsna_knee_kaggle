"""mavitgold — INSTRUMENT, never a submission: hengck23's MaViT (MIT) scored on the 58 organiser-gold studies.

knee-1oxsy names a fourth, decorrelated producer as the only lever left that could clear 0.957. The public scan
found one candidate, hengck23's MALA-T reader (`hengck23/hengck23-rnsa-knee`, `00000008.pth`), with no public
score and no stated training data. Its own notebook evaluates on `train_golden58.csv`, so gold58 is at least the
set its author held out to read it; that is the premise this reading rests on, and it is not proven.

Writes `mavit_gold58.csv` (per-study probabilities) and prints per-label AUC, macro AUC and seconds per study.
Loads the foreign checkpoint with `weights_only=True`. Pre-registered read against our gold58 macro (~0.90
midpoint for `band_dense_hold`): >= 0.89 earns a board seat at a small share beside S6 at stride 2;
0.85-0.89 earns the rank-correlation read first; < 0.85 closes it as a seat.
"""

import logging
import sys
import time
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)
    root = Path("/kaggle/input")
    checkpoint = next(
        path for pattern in ("*/00000008.pth", "*/*/00000008.pth", "*/*/*/00000008.pth") for path in root.glob(pattern)
    )
    competition = next(path.parent for pattern in ("*/train_series", "*/*/train_series") for path in root.glob(pattern))
    sys.path.append(str(checkpoint.parent))

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # pyrefly: ignore[missing-import] - hengck23's source, on sys.path only from here
    from model import KneeModel  # noqa: PLC0415

    # pyrefly: ignore[missing-import]
    from my_fastread_knee import make_knee_dataset_reader  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    labels = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]
    studies = pd.read_csv(checkpoint.parent / "train_golden58.csv")
    series = pd.read_csv(checkpoint.parent / "train_golden58_series.csv")
    model = KneeModel(pretrained=False)
    saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
    log.info("resume: %s", model.load_state_dict(saved["model_state_dict"], strict=False))
    model = model.to("cuda").eval()
    reader = make_knee_dataset_reader(
        out_size=288, num_slice=8, percentile=(1.0, 99.0), crop_mm=130.0, io_threads=32, crop_only_if_nonsquare=True
    )
    rows = []
    started = time.time()
    for study in studies["StudyInstanceUID"]:
        read = reader(study, series[series["StudyInstanceUID"] == study], str(competition / "train_series"), "valid")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logit = model.infer(
                torch.from_numpy(read["volume"]).unsqueeze(0).byte().cuda(),
                torch.from_numpy(read["view_mask"]).unsqueeze(0).bool().cuda(),
                torch.from_numpy(read["slice"]).unsqueeze(0).float().cuda(),
            )
        rows.append(torch.sigmoid(logit).float().cpu().numpy()[0])
    log.info("seconds per study %.2f", (time.time() - started) / len(studies))
    predicted = pd.DataFrame(np.stack(rows), columns=labels, index=studies["StudyInstanceUID"])
    predicted.to_csv("/kaggle/working/mavit_gold58.csv")
    areas = {label: roc_auc_score(studies[label].to_numpy(), predicted[label].to_numpy()) for label in labels}
    for label, area in areas.items():
        log.info("%-18s %.4f", label, area)
    log.info("MACRO %.4f", float(np.mean(np.asarray(list(areas.values()), dtype=float))))


if __name__ == "__main__":
    main()
