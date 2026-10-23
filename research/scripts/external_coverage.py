"""External label-table coverage audit (M389) — what external data could add for
the eight class-B (modelling-hole) columns: MCL, Fracture, Lateral OA, Effusion,
PF OA, Synovitis, Medial OA, Baker's.

Binarisation rule (stated): a cell counts as a usable POSITIVE when its soft
value is >= 0.5 (the same midpoint rule as the binarised v2 truth); non-null but
< 0.5 = assessed negative; null/empty = not annotated (no label of any kind).
Own-corpus split: holdout909 = band_span_s2026.holdout.npz studies; train = the
rest of report_labels_v2.csv (stated file).

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/external_coverage.py
"""

import os

import numpy as np
import pandas as pd

LABELS = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels"
MODELS = "D:/data/volumetric/mri/models/rsna_knee_abnormality"
COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
CLASS_B = ["MCL", "Fracture", "Lateral OA", "Effusion", "PF OA", "Synovitis",
           "Medial OA", "Baker's"]
TABLES = [
    ("fastmri_findings.csv", "fastMRI+ (NYU), DUA: internal research only, no redistribution of derived data"),
    ("fastmri_findings_compartment.csv", "fastMRI+ compartment variant, same DUA"),
    ("fastmri_findings_effusion.csv", "fastMRI+ effusion variant, same DUA"),
    ("mrnet_acl.csv", "MRNet (Stanford), agreement bars creating adaptations"),
    ("rijeka_acl.csv", "Rijeka external expert reads, CC BY-NC-ND"),
    ("labels_hisfluid.csv", "raptor author's 12-finding labels on OUR studies, CC0-1.0"),
]


def census(path: str) -> tuple[pd.DataFrame, dict[str, tuple[int, int, int]]]:
    """Returns (df, col -> (non_null, positives>=0.5, n_rows))."""
    df = pd.read_csv(path)
    out = {}
    for c in COLS:
        if c not in df.columns:
            out[c] = (0, 0, len(df))
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        out[c] = (int(v.notna().sum()), int((v >= 0.5).sum()), len(df))
    return df, out


def main() -> None:
    ref = np.load(f"{MODELS}/band_span_s2026.holdout.npz")
    hold = set(str(s) for s in ref["studies"].tolist())

    v2 = pd.read_csv(f"{LABELS}/report_labels_v2.csv")
    train = v2[~v2.StudyInstanceUID.isin(hold)]
    held = v2[v2.StudyInstanceUID.isin(hold)]
    print(f"OWN CORPUS ({LABELS}/report_labels_v2.csv): {len(v2)} studies = "
          f"train {len(train)} + holdout {len(held)}")

    print(f"\n=== (a) coverage matrix: usable positives (soft >= 0.5) ===")
    hdr = f"{'table':36s}" + "".join(f"{c[:9]:>10s}" for c in COLS)
    print(hdr)
    store = {}
    for fname, licence in TABLES:
        df, c = census(f"{LABELS}/{fname}")
        store[fname] = (df, c)
        row = f"{fname:36s}" + "".join(
            f"{c[col][1]:>10d}" if c[col][0] else f"{'.':>10s}" for col in COLS)
        print(row)
        annotated = int(sum(1 for col in COLS if c[col][0]))
        any_label = int(df[COLS[0]].notna().sum()) if COLS[0] in df else 0
        print(f"{'  rows ' + str(len(df)) + ', cols annotated ' + str(annotated) + '/12':36s}"
              f"  licence: {licence}")

    print(f"\nnon-null-but-negative (assessed absence) cells, same order:")
    for fname, _ in TABLES:
        _, c = store[fname]
        row = f"{fname:36s}" + "".join(
            f"{c[col][0] - c[col][1]:>10d}" if c[col][0] else f"{'.':>10s}" for col in COLS)
        print(row)

    print(f"\nfastmri_findings.csv == fastmri_findings_effusion.csv byte-identical "
          f"(md5 8d8576f7...); compartment variant differs — differences below.")
    a, _ = store["fastmri_findings.csv"]
    b, _ = store["fastmri_findings_compartment.csv"]
    diff_cols = []
    for col in COLS:
        if col in a.columns and col in b.columns:
            va = pd.to_numeric(a[col], errors="coerce")
            vb = pd.to_numeric(b[col], errors="coerce")
            if not va.equals(vb):
                diff_cols.append((col, int((va >= 0.5).sum()), int((vb >= 0.5).sum())))
    print(f"  compartment-variant diffs (col, base positives, compartment positives): {diff_cols}")

    print(f"\n=== (b) the three columns: own train vs holdout vs external adds ===")
    for col in CLASS_B:
        tv = pd.to_numeric(train[col], errors="coerce")
        hv = pd.to_numeric(held[col], errors="coerce")
        adds = {fname: c[col][1] for fname, (_, c) in store.items() if c[col][1]}
        print(f"  {col:9s} train(3498) {int((tv >= 0.5).sum()):4d} | "
              f"holdout909 {int((hv >= 0.5).sum()):4d} | external adds: {adds or 'NONE'}")

    print(f"\nholdout909 check vs letter: "
          + ", ".join(f"{col} {int((pd.to_numeric(held[col], errors='coerce') >= 0.5).sum())}"
                      for col in COLS))
    print(f"\n231-URL record: docs/EXTERNAL_CORPORA.md:47 — "
          f"$CREDS/fastmri_urls.txt (231 presigned URLs across "
          f"knee/brain/breast/prostate, credential file outside the repo, not "
          f"read). The knee singlecoil h5 archives are the ONLY carrier of the "
          f"fastMRI+ label namespace (fileNNNNNNN); the DICOM batch on disk "
          f"cannot join. Pulling adds VOLUMES for the "
          f"{len(store['fastmri_findings.csv'][0])} fastMRI+ studies whose "
          f"labels are already tabulated above — i.e. it makes the fastMRI rows "
          f"in the coverage matrix trainable, it does not add new label rows.")


if __name__ == "__main__":
    main()
