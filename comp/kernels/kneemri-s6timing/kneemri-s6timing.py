"""INSTRUMENT, never an entry: S6's cost per study at eval stride 3 and 2, and how far its predictions move.

S6 scores window centres at every third slice (`compact_infer --eval-stride 3`, the training-time validation
default). Every slice is in-distribution — training sampled any centre — so a denser stride is a test-time
average with no new input. The hidden test is ~1,322 studies against a 3-study visible run, and S6's deadline
degrades overrunning studies to the column mean (`--max-fallback-fraction 1.0`), so a denser stride must be
timed before it is shipped (knee-1oxsy). Both passes run S6's own runner on the SAME first N train studies.
S6 trained on the whole pool, so these predictions are in-sample: they price TIME and SHIFT, never a score.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

STUDIES = 120
STRIDES = (3, 2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)
    working = Path("/kaggle/working")
    found = sorted(Path("/kaggle/input").rglob("helios_s6.py"))
    if len(found) != 1:
        message = f"expected one helios_s6.py under /kaggle/input, found {found}"
        raise RuntimeError(message)
    original = found[0].read_text(encoding="utf-8")
    probe = Path("/kaggle/temp/probe") if Path("/kaggle/temp").is_dir() else Path("/tmp/probe")
    summary = {}
    for stride in STRIDES:
        patches = {
            "'--split', 'test',": f"'--split', 'train', '--limit', '{STUDIES}', '--max-fallback-fraction', '1.0',",
            "str(manifest.get('eval_stride', 3))": f"'{stride}'",
            "if rc != 0:": "if rc != 0 and not (pred / 'submission.csv').is_file():",
            "shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')": "pass",
            "receipt = json.loads((pred / 'submission_receipt.json').read_text())": "raise SystemExit(0)",
        }
        source = original
        for anchor, replacement in patches.items():
            if source.count(anchor) != 1:
                message = f"S6 source holds {source.count(anchor)} copies of {anchor!r}, expected one"
                raise ValueError(message)
            source = source.replace(anchor, replacement)
        script = Path(f"/tmp/helios_s6_stride{stride}.py")
        script.write_text(source, encoding="utf-8")
        shutil.rmtree(probe / "prediction", ignore_errors=True)
        started = time.time()
        environment = dict(os.environ, S6_T0=str(started))
        finished = subprocess.run([sys.executable, "-u", str(script)], check=False, cwd=working, env=environment)  # noqa: S603 - our mounted copy, patched, fixed argv
        spent = time.time() - started
        out = working / f"stride{stride}"
        out.mkdir(exist_ok=True)
        for worker in sorted((probe / "prediction").glob("worker*")):
            for name in ("log.txt", "predictions.csv", "receipt.json"):
                if (worker / name).is_file():
                    shutil.copyfile(worker / name, out / f"{worker.name}_{name}")
        summary[stride] = {"exit": finished.returncode, "seconds": round(spent, 1)}
        log.info("S6 stride %d exited %d after %.0fs", stride, finished.returncode, spent)
    (working / "timing.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("TIMING %s", json.dumps(summary))


if __name__ == "__main__":
    main()
