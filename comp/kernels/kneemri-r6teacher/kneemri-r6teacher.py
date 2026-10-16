"""INSTRUMENT, never an entry: heliosli's R6 bundle `fe055406` with flip TTA, scored on every TRAIN study.

The teacher for knee-p844l. `nartaa_soup_k4` distilled nartaa's CoAtNet alone into our soup and read solo 0.948
against 0.946; the pf line's teacher is that CoAtNet plus this R6 reader, and the R6 half has never been scored
on train because its runner may not run off Kaggle. This is `kneemri-s6r7pfheavy`'s R7 stage with the split moved
to train and the deadline to 11 h, keeping each worker's `predictions.csv`. R6 trained on the whole pool, so these
predictions are in-sample, as the nartaa teacher's were: a distillation target, never a score.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEADLINE_HOURS = 11.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)
    working = Path("/kaggle/working")
    found = sorted(Path("/kaggle/input").rglob("helios_r7.py"))
    if len(found) != 1:
        message = f"expected one helios_r7.py under /kaggle/input, found {found}"
        raise RuntimeError(message)
    patches = {
        "if manifest.get('flip_tta'):": "if True:",
        "BUNDLE_ID = '8c7b666e27dc0e8f'": "BUNDLE_ID = 'fe055406fad6a12f'",
        "C_SLUG = 'rsna-knee-compact-r7cnxfsss'": "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'",
        "DEADLINE_HOURS = 8.3": f"DEADLINE_HOURS = {DEADLINE_HOURS}",
        "'--split', 'test',": "'--split', 'train',",
        "if rc != 0:": "if rc != 0 and not any(pred.glob('worker*/predictions.csv')):",
        "shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')": "pass",
        "receipt = json.loads((pred / 'submission_receipt.json').read_text())": "raise SystemExit(0)",
    }
    source = found[0].read_text(encoding="utf-8")
    for anchor, replacement in patches.items():
        if source.count(anchor) != 1:
            message = f"R7 source holds {source.count(anchor)} copies of {anchor!r}, expected one"
            raise ValueError(message)
        source = source.replace(anchor, replacement)
    script = Path("/tmp/helios_r6_teacher.py")
    script.write_text(source, encoding="utf-8")
    started = time.time()
    environment = dict(os.environ, S6_T0=str(started))
    finished = subprocess.run([sys.executable, "-u", str(script)], check=False, cwd=working, env=environment)  # noqa: S603 - our mounted copy, patched, fixed argv
    spent = time.time() - started
    probe = Path("/kaggle/temp/probe") if Path("/kaggle/temp").is_dir() else Path("/tmp/probe")
    out = working / "r6teacher"
    out.mkdir(exist_ok=True)
    harvested = []
    for worker in sorted((probe / "prediction").glob("worker*")):
        for name in ("log.txt", "predictions.csv", "receipt.json"):
            if (worker / name).is_file():
                shutil.copyfile(worker / name, out / f"{worker.name}_{name}")
                harvested.append(f"{worker.name}_{name}")
    summary = {"exit": finished.returncode, "seconds": round(spent, 1), "harvested": harvested}
    (working / "teacher.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("TEACHER %s", json.dumps(summary))
    if not any(name.endswith("predictions.csv") for name in harvested):
        message = f"R6 left no predictions (exit {finished.returncode}), so there is no teacher"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
