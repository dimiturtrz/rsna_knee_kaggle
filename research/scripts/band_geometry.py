"""Band-geometry falsification test (M390): are the class-B columns peripheral
anatomy that band cropping discards?

Class-B (modelling holes): MCL, Lateral OA, Fracture, Effusion, PF OA,
Synovitis, Medial OA, Baker's.  Class-C (label-limited, central): Medial
Meniscus, Lateral Meniscus, ACL, Contusion.

Cut: per-column AUC @holdout909, midpoint-binarised v2 truth, honest pool
(void foreign out). Band source: the metrics.jsonl recipe's RESOLVED
band_low/band_high (the recipe does not record whether the flag was passed or
defaulted — stated); name-suffix _n12b0.1-0.9 / _n24b0.1-0.9 for the
decode-time re-banded variants (same checkpoint, re-decoded holdout); unknown
otherwise and counted.

Clean contrasts (recipe identity verified in-script):
  trained triple, 12 slices/336/v2/e12/max, ONLY band differs:
    lr4e-5_ws_s2026 (0.2-0.8) / band_span_s2026 (0.1-0.9) / band_full_s2026 (0.0-1.0)
  same-weights decode pairs: lr4e-5_ws vs its _n12b0.1-0.9 variant;
    merged_blend_cf vs its _n12b0.1-0.9 variant (licence-tainted, marked)
  slices control: band_dense_s2026 vs _n24b0.1-0.9 (band FIXED 0.1-0.9, 20->24 slices)

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/band_geometry.py
"""

import glob
import json
import os
import re
import statistics as st

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/models/rsna_knee_abnormality"
META = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/metadata/canonical_metadata.csv"
COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
CLASS_B = ["MCL", "Lateral OA", "Fracture", "Effusion", "PF OA", "Synovitis",
           "Medial OA", "Baker's"]
CLASS_C = ["Medial Meniscus", "Lateral Meniscus", "ACL", "Contusion"]
VOID = {
    "raptor_v4_swa", "raptor_v4", "raptor_v9", "raptor_effv2l480",
    "raptor_cnv2b336", "raptor_coat384", "dinov3f0_solo",
    "foreign_dinov3_matched", "foreign_dinov3_band1189",
    "foreign_dinov3_band1387", "foreign_dinov3_band1486",
    "foreign_dinov3_band1090", "foreign_dinov3_tta4", "foreign_dinov3_tta3",
    "foreign_dinov3_band1684", "foreign_dinov3_band0892",
    "foreign_dinov3_band208",
}
TRIPLE = ["lr4e-5_ws_s2026", "band_span_s2026", "band_full_s2026"]
REBANDED = [("lr4e-5_ws_s2026", "lr4e-5_ws_s2026_n12b0.1-0.9"),
            ("merged_blend_cf", "merged_blend_cf_n12b0.1-0.9")]
SLICES_CTRL = ("band_dense_s2026", "band_dense_s2026_n24b0.1-0.9")


def per_col_auc(t, s):
    ranks = rankdata(s, axis=0)
    out = np.empty(12)
    for j in range(12):
        pos = t[:, j] == 1
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        out[j] = (ranks[pos, j].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def band_of(name, recipe):
    if recipe is not None:
        try:
            return (float(recipe["band_low"]), float(recipe["band_high"]))
        except (KeyError, TypeError, ValueError):
            pass
    m = re.search(r"_n\d+b([\d.]+)-([\d.]+)$", name)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def main() -> None:
    print("TASK 0 — the default: SLICE_BAND = (0.20, 0.80), "
          "core/data/corpus/slot_reader.py:35; TrainingOptions.band_low/band_high "
          "default to it (kneemri/training/options.py:104-105). The default is "
          "NOT full-stack, and the slot_reader docstring records the axis as "
          "closed at 12 slices: 0.2-0.8 = 0.8525, 0.1-0.9 = 0.8565, "
          "0.0-1.0 = 0.8427 (interior optimum; full stack REFUTED). Every "
          "shipped member passes 0.1/0.9 explicitly. The recipe records the "
          "RESOLVED band, so a passed-vs-defaulted split is not recoverable — "
          "there is no separable no-flag group.")

    ref = np.load(f"{ROOT}/band_span_s2026.holdout.npz")
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)

    recipes = {}
    for p in glob.glob(f"{ROOT}/*.metrics.jsonl"):
        name = os.path.basename(p)[:-len(".metrics.jsonl")]
        try:
            recipes[name] = json.loads(open(p).readline())["recipe"]
        except (json.JSONDecodeError, KeyError):
            pass

    aucs, bands = {}, {}
    n_skip = 0
    for p in sorted(glob.glob(f"{ROOT}/*.holdout.npz")):
        name = os.path.basename(p)[:-len(".holdout.npz")]
        if name in VOID:
            continue
        z = np.load(p)
        st_ = [str(s) for s in z["studies"].tolist()]
        if set(st_) != set(ref_studies) or len(st_) != 909:
            n_skip += 1
            continue
        order = np.array([pos[s] for s in st_])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.allclose(tr, np.asarray(ref["truth"], float)):
            n_skip += 1
            continue
        aucs[name] = per_col_auc(t0, pr)
        bands[name] = band_of(name, recipes.get(name))

    unknown = sorted(n for n, b in bands.items() if b is None)
    print(f"\nPOOL: honest n={len(aucs)} @holdout909-midpoint-v2 "
          f"(skipped {n_skip} non-909/truth-mismatch; void foreign excluded). "
          f"Band undetermined for {len(unknown)} members "
          f"(name-inferred pool: probes, run1s, foreign-named honest, fold0 bases).")

    groups = {}
    for n, b in bands.items():
        if b is None:
            continue
        groups.setdefault(b, []).append(n)
    print(f"\n=== (a) per-column median AUC by band group ===")
    hdr = f"{'band':>10s} {'n':>3s} " + " ".join(f"{c[:8]:>9s}" for c in COLS)
    print(hdr)
    for b in sorted(groups):
        mem = groups[b]
        M = np.median(np.array([aucs[n] for n in mem]), axis=0)
        print(f"{str(b):>10s} {len(mem):3d} " + " ".join(f"{v:9.4f}" for v in M))
    bmean = {}
    for b in sorted(groups):
        M = np.median(np.array([aucs[n] for n in groups[b]]), axis=0)
        bmean[b] = (np.mean([M[COLS.index(c)] for c in CLASS_B]),
                    np.mean([M[COLS.index(c)] for c in CLASS_C]))
    print("class means by group (B = 8 peripheral-hypothesis cols, C = 4 central):")
    for b, (mb, mc) in bmean.items():
        print(f"  {str(b):>10s} n={len(groups[b]):3d}  class-B {mb:.4f}  class-C {mc:.4f}")

    def recipe_diff(x, y):
        a, c = recipes[x], recipes[y]
        return {k: (a.get(k), c.get(k)) for k in sorted(set(a) | set(c))
                if k not in ("out", "paths", "slice_regions") and a.get(k) != c.get(k)}

    print(f"\n=== the GUARD: clean band-only contrasts ===")
    d = recipe_diff("band_span_s2026", "band_full_s2026")
    d2 = recipe_diff("band_span_s2026", "lr4e-5_ws_s2026")
    print(f"  recipe diff band_span vs band_full: {d}")
    print(f"  recipe diff band_span vs lr4e-5_ws: {d2}")

    def contrast(lo, hi, note):
        dl = aucs[hi] - aucs[lo]
        print(f"\n  {note}: {lo} -> {hi}")
        for j, c in enumerate(COLS):
            mark = "B" if c in CLASS_B else "C"
            print(f"    [{mark}] {c:17s} {aucs[lo][j]:.4f} -> {aucs[hi][j]:.4f}  d={dl[j]:+.4f}")
        mb = float(np.mean([dl[COLS.index(c)] for c in CLASS_B]))
        mc = float(np.mean([dl[COLS.index(c)] for c in CLASS_C]))
        print(f"    class-B mean delta {mb:+.4f} | class-C mean delta {mc:+.4f}")
        return mb, mc

    contrast("lr4e-5_ws_s2026", "band_span_s2026",
             "TRAINED 0.2-0.8 -> 0.1-0.9 (identical recipes, 12sl)")
    contrast("band_span_s2026", "band_full_s2026",
             "TRAINED 0.1-0.9 -> 0.0-1.0 (identical recipes, 12sl)")
    for lo, hi in REBANDED:
        lic = " [LICENCE-TAINTED]" if "merged" in lo else ""
        contrast(lo, hi, f"SAME WEIGHTS, decode-time re-band 0.2-0.8 -> 0.1-0.9{lic}")
    lo, hi = SLICES_CTRL
    contrast(lo, hi, "SLICES CONTROL: band FIXED 0.1-0.9, 20 -> 24 slices "
                     "(separates band from slice count)")

    print(f"\n=== (b) sharpest single contrast: band_full (0.0-1.0) vs "
          f"band_span (0.1-0.9), identical recipes ===")
    for c in ("MCL", "Lateral OA"):
        j = COLS.index(c)
        print(f"  {c:11s} span {aucs['band_span_s2026'][j]:.4f} vs full "
              f"{aucs['band_full_s2026'][j]:.4f}  (full-span {aucs['band_full_s2026'][j] - aucs['band_span_s2026'][j]:+.4f})")
        print(f"  {c:11s} span {aucs['band_span_s2026'][j]:.4f} vs ws0.2-0.8 "
              f"{aucs['lr4e-5_ws_s2026'][j]:.4f}  (span-ws {aucs['band_span_s2026'][j] - aucs['lr4e-5_ws_s2026'][j]:+.4f})")

    print(f"\n=== (c) slice-axis orientation ===")
    m = pd.read_csv(META)
    print(f"  canonical_metadata.csv: {len(m)} studies; plane value counts:")
    for v, n in m.plane.value_counts().items():
        print(f"    {v}: {n}")
    print(f"  sequence mix: {dict(m.sequence.value_counts().head(4))}")
    print(f"  field strength: {dict(m.field_strength_t.value_counts().head(3))}")
    print("  slot_reader.band_indices samples WITHIN one series' ordered file "
          "list; every study is Axial+Coronal+Sagittal (4407/4407 — uniform, "
          "no per-study plane variation). Slice-axis meaning differs by plane: "
          "sagittal edges = medial/lateral joint line, coronal edges = "
          "anterior/posterior, axial edges = superior/inferior — so one band "
          "crop trims DIFFERENT anatomy per plane, uniformly across studies. "
          "From disk I cannot establish per-series slice counts per study, but "
          "the plane mix is exact and uniform.")


if __name__ == "__main__":
    main()
