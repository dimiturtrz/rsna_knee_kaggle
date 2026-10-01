"""Rank every solo member by holdout AUC @holdout909, midpoint cut, v2 ruler (T0),
attach provenance attributes from metrics.jsonl recipes (with parent-stem
inheritance and a curated foreign map), and answer four questions about the
attribute space.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/solo_rank.py
"""

import csv
import glob
import json
import os
import re
import statistics as st

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = "D:/data/volumetric/mri/models/rsna_knee_abnormality"
REF = os.path.join(ROOT, "band_span_s2026.holdout.npz")
COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

VOID = {
    "raptor_v4_swa", "raptor_v4", "raptor_v9", "raptor_effv2l480",
    "raptor_cnv2b336", "raptor_coat384", "dinov3f0_solo",
    "foreign_dinov3_matched", "foreign_dinov3_band1189",
    "foreign_dinov3_band1387", "foreign_dinov3_band1486",
    "foreign_dinov3_band1090", "foreign_dinov3_tta4", "foreign_dinov3_tta3",
    "foreign_dinov3_band1684", "foreign_dinov3_band0892",
    "foreign_dinov3_band208",
}
LICENCE = {  # rijeka / mrnet / fastmri provenance (roster tags + declines)
    "merged_blend_cf", "merged_blend_s2026", "merged_dense_cf", "merged_span_cf",
    "merged_corpus_s2026", "merged3_corpus_s2026", "merged_distill_cf",
    "merged_labelwise_cf", "merged_blend_cf_r224", "merged_blend_cf_r280",
    "merged_blend_cf_r392", "merged_blend_cf_n12b0.1-0.9", "merged_dense_cf_run1",
    "merged3_oversample_s2026", "ortho_fastmri_s2026", "ortho_fastmri_masked_s2026",
    "mrnet_warm_s2026", "mrnet_warm_s2026_run1",
}

FAMILY_BY_VARIANT = {
    "small": ("dinov2", "small"), "base": ("dinov2", "base"),
    "v3small": ("dinov3", "small"), "v3splus": ("dinov3", "small+"),
    "v3base": ("dinov3", "base"), "v3large": ("dinov3", "large"),
    "ortholarge": ("orthodiffusion", "large"), "swinbase": ("swin", "base"),
    "cnxbase": ("convnextv2", "base"), "resnet50d": ("resnet", "50d"),
    "vjepa2l": ("vjepa2", "large"), "coatnet2": ("coatnet", "v2"),
    "mricore": ("mricore", "?"), "raddino": ("radiodino", "b16"),
    "radnet121": ("radnet", "121"), "medsam": ("medsam", "vit-b"),
}
NAME_FAMILY = [
    (r"^raptor_effv2", ("efficientnet", "v2l")), (r"^raptor_cnv2", ("convnextv2", "base")),
    (r"^raptor_coat", ("coatnet", "?")), (r"^raptor", ("raptor", "v4/coatnet")),
    (r"^foreign_dinov3|^dinov3f0", ("dinov3", "served-folds")),
    (r"^ortho", ("orthodiffusion", "?")), (r"^radiodino|^raddino", ("radiodino", "?")),
    (r"^radnet", ("radnet", "?")), (r"^mricore", ("mricore", "?")),
    (r"^swin", ("swin", "?")), (r"^cnx", ("convnextv2", "?")),
    (r"^resnet|^res392", ("resnet", "?")), (r"^vjepa", ("vjepa2", "?")),
    (r"^medsam", ("medsam", "?")), (r"^vit_in21k", ("vit", "in21k")),
    (r"^v3base", ("dinov3", "base")), (r"^v3large", ("dinov3", "large")),
    (r"^v3small|lora_r8_v3small|^errors_v3small", ("dinov3", "small")),
    (r"^dinov3_base", ("dinov3", "base")), (r"^dinov3_v3", ("dinov3", "small")),
    (r"^base|^blend_base", ("dinov2", "base")),
]
MEDICAL_PRETRAIN_FAMILIES = {"orthodiffusion", "radiodino", "mricore", "medsam", "radnet"}


def macro_auc(t, p):
    return float(np.mean([roc_auc_score(t[:, j], p[:, j]) for j in range(12)]))


def parent_stem(name):
    s = re.sub(r"_run\d+$", "", name)
    s = re.sub(r"_r(224|280|392)$", "", s)
    s = re.sub(r"_n\d+b[\d.]+-[\d.]+$", "", s)
    return s


def label_family(basename, scheme):
    b = basename.lower()
    if scheme == "mergedf" or "mergedf" in b: return "mergedf-fastmri"
    if scheme == "merged3" or "merged3" in b: return "merged3-mrnet"
    if scheme == "merged" or b.startswith("merged"): return "merged-rijeka"
    if "mrnet" in b: return "mrnet"
    if "blend_three" in b: return "blend-three-llm"
    if "blend_hybrid" in b or "v4hybrid" in b: return "blend-hybrid"
    if "corrected" in b: return "v2-corrected"
    if "delonly" in b: return "v2-delonly"
    if "v2" in b: return "v2"
    if b.startswith("report_labels_"): return b.replace("report_labels_", "").replace(".csv", "")
    return b.replace(".csv", "") or "?"


def build_attributes(name, recipe, inferred):
    a = {"family": "?", "size": "?", "res": "?", "slices": "?", "span": "?",
         "epochs": "?", "labels": "?", "warm": "stock", "head": "?",
         "loss": set(), "external": "none"}
    if recipe:
        fam, size = FAMILY_BY_VARIANT.get(str(recipe.get("variant")), ("?", "?"))
        a["family"], a["size"] = fam, size
        a["res"] = str(recipe.get("image_size"))
        a["slices"] = str(recipe.get("slices"))
        try:
            a["span"] = f"{float(recipe['band_high']) - float(recipe['band_low']):.2f}"
        except (TypeError, KeyError):
            pass
        a["epochs"] = str(recipe.get("epochs"))
        a["labels"] = label_family(os.path.basename(str(recipe.get("labels", ""))),
                                   str(recipe.get("scheme")))
        a["head"] = f"{recipe.get('slice_pooling')}/{recipe.get('pool')}"
        if str(recipe.get("grade_ladder")) == "True": a["loss"].add("grade_ladder")
        if str(recipe.get("per_label_loss")) == "True": a["loss"].add("per_label")
        if str(recipe.get("focus_label")) not in ("None", ""): a["loss"].add("focus")
        if float(recipe.get("distill_weight", 0) or 0) > 0: a["loss"].add("distill")
        if float(recipe.get("mixup", 0) or 0) > 0: a["loss"].add("mixup")
        if str(recipe.get("severity_weight")) == "True": a["loss"].add("severity")
        if str(recipe.get("calibrated_targets")) == "True": a["loss"].add("calib-tgt")
        if str(recipe.get("hold_annotations")) == "True": a["loss"].add("hold-annot")
        if str(recipe.get("ema")) == "True": a["loss"].add("ema")
        init, ew = str(recipe.get("init")), str(recipe.get("encoder_weights", "None"))
        if "mrnet" in init: a["warm"] = "mrnet-pretrain"
        elif "adapt" in ew or "pretrain" in ew or "latent" in ew:
            a["warm"] = "own-corpus-mim"
        elif "raptor" in ew: a["warm"] = "raptor-features"
        if a["labels"].startswith(("merged", "mrnet")):
            a["external"] = "extra-corpus:" + a["labels"]
        elif a["warm"] == "mrnet-pretrain" or a["family"] in MEDICAL_PRETRAIN_FAMILIES:
            a["external"] = "medical-pretrain"
    else:
        inferred.add(name)
        for pat, (fam, size) in NAME_FAMILY:
            if re.search(pat, name):
                a["family"], a["size"] = fam, size
                break
        else:
            a["family"], a["size"] = "dinov2", "small"  # pool monoculture default
        m = re.search(r"(?<!\d)(160|224|280|336|392|480)(?!\d)", name)
        if m: a["res"] = m.group(1)
        m = re.search(r"(?:slices|depth|s)(\d{2})(?!\d)", name)
        if m: a["slices"] = m.group(1)
        if "@focal" in name or name.endswith("_probe") or "_head" in name:
            a["head"] = "feature-probe"
        if name.startswith(("raptor", "foreign_dinov3", "dinov3f0", "foreign_oof")):
            a["external"] = "foreign-weights"
        elif a["family"] in MEDICAL_PRETRAIN_FAMILIES:
            a["external"] = "medical-pretrain"
        a["labels"] = "v2?" if a["external"] == "none" else "?"
    return a


def main():
    ref = np.load(REF)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)

    recipes, old_schema = {}, []
    for p in glob.glob(os.path.join(ROOT, "*.metrics.jsonl")):
        name = os.path.basename(p)[:-len(".metrics.jsonl")]
        try:
            recipes[name] = json.loads(open(p).readline())["recipe"]
        except (json.JSONDecodeError, KeyError):
            old_schema.append(name)

    scores, excl_studies, excl_truth = {}, [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "*.holdout.npz"))):
        name = os.path.basename(p)[:-len(".holdout.npz")]
        z = np.load(p)
        st_ = [str(s) for s in z["studies"].tolist()]
        if set(st_) != set(ref_studies) or len(st_) != 909:
            excl_studies.append(name)
            continue
        order = np.array([pos[s] for s in st_])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.allclose(tr, np.asarray(ref["truth"], float)):
            excl_truth.append(name)
            continue
        scores[name] = macro_auc(t0, pr)

    attrs, no_recipe = {}, set()
    for name in scores:
        rec = recipes.get(name) or recipes.get(parent_stem(name))
        attrs[name] = build_attributes(name, rec, no_recipe)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    n = len(ranked)
    vals = [s for _, s in ranked]
    honest = [(nm, s) for nm, s in ranked if nm not in VOID]
    hvals = [s for _, s in honest]
    med = st.median(vals)
    print(f"POPULATION: npz scanned {len(scores) + len(excl_studies) + len(excl_truth)}; "
          f"scored @holdout909-midpoint-v2 {n} "
          f"(excluded: {len(excl_studies)} wrong study set, {len(excl_truth)} truth mismatch); "
          f"metrics-only members with no 909 npz (dark/ff/cf/smoke): "
          f"{len(set(recipes) | set(old_schema)) - len(set(scores) & (set(recipes) | set(old_schema)))}; "
          f"of the scored, {len(VOID & set(scores))} VOID in-sample foreign, "
          f"{len(LICENCE & set(scores))} licence-tainted; "
          f"attributes from recipe for {n - len(no_recipe)}, name-inferred for {len(no_recipe)}")
    print(f"anchor: labelwise_depth24_s2026 = {scores['labelwise_depth24_s2026']:.4f} "
          f"(reference 0.8617)")

    print("\n=== TOP 30 (score, family/size, res, slices, span, epochs, labels, warm, head, loss, external) ===")
    for i, (nm, s) in enumerate(ranked[:30], 1):
        a = attrs[nm]
        flag = (" VOID" if nm in VOID else "") + (" LIC" if nm in LICENCE else "")
        print(f"{i:2d}. {s:.4f} {nm:34s} {a['family']}/{a['size']:12s} {a['res']:>3s}px "
              f"{a['slices']:>2s}sl sp{a['span']:4s} e{a['epochs']:>2s} {a['labels']:16s} "
              f"{a['warm']:16s} {a['head']:14s} {'+'.join(sorted(a['loss'])) or '-':28s} {a['external']}{flag}")
    print(f"... median context (rank {n//2-1}-{n//2+2} of {n}):")
    for i in range(n // 2 - 1, n // 2 + 3):
        nm, s = ranked[i - 1]
        print(f"{i:2d}. {s:.4f} {nm:34s} {attrs[nm]['family']}/{attrs[nm]['size']}")

    print(f"\n=== SPREAD (all {n}; honest {len(honest)}) ===")
    for q, v in [("best", vals[0]), ("p90", float(np.percentile(vals, 90))),
                 ("p75", float(np.percentile(vals, 75))), ("median", med),
                 ("p25", float(np.percentile(vals, 25))), ("p10", float(np.percentile(vals, 10))),
                 ("worst", vals[-1])]:
        print(f"  {q:7s} {v:.4f}")
    print(f"  gap best->median {vals[0] - med:+.4f} | honest best {hvals[0]:.4f} "
          f"({honest[0][0]}), honest median {st.median(hvals):.4f}, "
          f"gap {hvals[0] - st.median(hvals):+.4f}")

    pool = [(nm, s) for nm, s in ranked if nm not in VOID]
    print(f"\n=== ATTRIBUTE CONTRASTS (honest pool n={len(pool)}; median with vs without) ===")
    tests = [
        ("family dinov2-small", lambda a: a["family"] == "dinov2" and a["size"] == "small"),
        ("family dinov2-base", lambda a: a["family"] == "dinov2" and a["size"] == "base"),
        ("family dinov3 (any)", lambda a: a["family"] == "dinov3"),
        ("family orthodiffusion", lambda a: a["family"] == "orthodiffusion"),
        ("family radiodino/mricore", lambda a: a["family"] in ("radiodino", "mricore")),
        ("family swin/convnext/resnet/coatnet", lambda a: a["family"] in ("swin", "convnextv2", "resnet", "coatnet")),
        ("family vjepa2", lambda a: a["family"] == "vjepa2"),
        ("res 336", lambda a: a["res"] == "336"),
        ("res 224", lambda a: a["res"] == "224"),
        ("res 392/480", lambda a: a["res"] in ("392", "480")),
        ("slices >=20", lambda a: a["slices"].isdigit() and int(a["slices"]) >= 20),
        ("slices 12", lambda a: a["slices"] == "12"),
        ("span >=0.70", lambda a: a["span"] != "?" and float(a["span"]) >= 0.70),
        ("labels blend-three-llm", lambda a: a["labels"] == "blend-three-llm"),
        ("labels merged-rijeka", lambda a: a["labels"] == "merged-rijeka"),
        ("labels v2-corrected", lambda a: a["labels"] == "v2-corrected"),
        ("labels v3+", lambda a: a["labels"].startswith(("v3", "v5", "v6", "v7", "v8"))),
        ("head labelwise", lambda a: a["head"].startswith("labelwise")),
        ("head max", lambda a: a["head"].startswith("max")),
        ("loss calib-tgt", lambda a: "calib-tgt" in a["loss"]),
        ("loss hold-annot", lambda a: "hold-annot" in a["loss"]),
        ("loss ema", lambda a: "ema" in a["loss"]),
        ("loss distill", lambda a: "distill" in a["loss"]),
        ("loss grade_ladder", lambda a: "grade_ladder" in a["loss"]),
        ("loss mixup", lambda a: "mixup" in a["loss"]),
        ("warm own-corpus-mim", lambda a: a["warm"] == "own-corpus-mim"),
        ("external any", lambda a: a["external"] != "none"),
        ("external extra-corpus", lambda a: a["external"].startswith("extra-corpus")),
        ("external medical-pretrain", lambda a: a["external"] == "medical-pretrain"),
        ("epochs >=24", lambda a: a["epochs"].isdigit() and int(a["epochs"]) >= 24),
    ]
    rows = []
    for label, f in tests:
        w = [s for nm, s in pool if f(attrs[nm])]
        wo = [s for nm, s in pool if not f(attrs[nm])]
        if len(w) >= 3 and len(wo) >= 3:
            rows.append((st.median(w) - st.median(wo), label, len(w), st.median(w), len(wo), st.median(wo)))
    for d, label, nw, mw, nwo, mwo in sorted(rows, key=lambda r: -abs(r[0])):
        print(f"  {label:34s} with {mw:.4f} (n={nw:3d}) vs without {mwo:.4f} (n={nwo:3d})  d={d:+.4f}")

    print(f"\n=== TOP-DECILE PROFILE (honest pool; decile = top {len(pool)//10}) ===")
    k = max(1, len(pool) // 10)
    top = {nm for nm, _ in pool[:k]}
    print(f"  top-decile floor: {pool[k - 1][1]:.4f}; members: "
          + ", ".join(nm for nm, _ in pool[:k]))
    for label, f in tests:
        intop = sum(1 for nm, _ in pool if nm in top and f(attrs[nm]))
        inall = sum(1 for nm, _ in pool if f(attrs[nm]))
        if inall >= 3 and intop / k >= 0.4 and intop / inall >= 1.6 * (k / len(pool)) * 10 / 10:
            print(f"  enriched: {label:32s} {intop}/{k} of decile vs {inall}/{len(pool)} of pool")

    print(f"\n=== EMPTY CELLS (honest pool; additive-marginal sketch, NOT a fit) ===")
    base = st.median([s for _, s in pool])
    marg = {label: d for d, label, _, _, _, _ in rows}
    fam_of = lambda a: ("dinov2-small" if a["family"] == "dinov2" and a["size"] == "small"
                        else "dinov2-base" if a["family"] == "dinov2"
                        else a["family"])
    head_of = lambda a: "labelwise" if a["head"].startswith("labelwise") else (
        "max" if a["head"].startswith("max") else "other")
    lab_of = lambda a: (a["labels"] if a["labels"] in
                        ("v2", "blend-three-llm", "v2-corrected", "merged-rijeka") else "other")
    sl_of = lambda a: ("20-24" if a["slices"].isdigit() and int(a["slices"]) >= 20
                       else "12-16" if a["slices"].isdigit() else "?")
    fams = ["dinov2-small", "dinov2-base", "dinov3", "orthodiffusion",
            "convnextv2", "swin", "resnet"]
    heads, labs, sls = ["max", "labelwise"], ["v2", "blend-three-llm", "v2-corrected"], ["12-16", "20-24"]
    seen = {(fam_of(attrs[nm]), head_of(attrs[nm]), lab_of(attrs[nm]), sl_of(attrs[nm]))
            for nm, _ in pool}
    fam_marg = {"dinov2-small": marg.get("family dinov2-small", 0.0),
                "dinov2-base": marg.get("family dinov2-base", 0.0),
                "dinov3": marg.get("family dinov3 (any)", 0.0),
                "orthodiffusion": marg.get("family orthodiffusion", 0.0),
                "convnextv2": marg.get("family swin/convnext/resnet/coatnet", 0.0),
                "swin": marg.get("family swin/convnext/resnet/coatnet", 0.0),
                "resnet": marg.get("family swin/convnext/resnet/coatnet", 0.0)}
    lab_marg = {"v2": 0.0, "blend-three-llm": marg.get("labels blend-three-llm", 0.0),
                "v2-corrected": marg.get("labels v2-corrected", 0.0)}
    head_marg = {"max": marg.get("head max", 0.0), "labelwise": marg.get("head labelwise", 0.0)}
    sl_marg = {"20-24": marg.get("slices >=20", 0.0), "12-16": 0.0}
    empty = []
    for f_ in fams:
        for h_ in heads:
            for l_ in labs:
                for s_ in sls:
                    if (f_, h_, l_, s_) not in seen:
                        est = base + fam_marg[f_] + head_marg[h_] + lab_marg[l_] + sl_marg[s_]
                        empty.append((est, f_, h_, l_, s_))
    print(f"  honest-pool median {base:.4f}, honest best {pool[0][1]:.4f}; "
          f"est = median + marginals (marginals are confounded — realised combos "
          f"over-add by ~2x, so read these as upper-bound sketches)")
    for est, f_, h_, l_, s_ in sorted(empty, key=lambda r: -r[0])[:12]:
        print(f"  est {est:.4f}  {f_:15s} {h_:9s} {l_:16s} slices {s_}")
    print(f"  (v2-corrected marginal exists at n=1 only — single trial "
          f"v2_corrected_s2026 = {scores.get('v2_corrected_s2026', float('nan')):.4f} "
          f"at a plain operating point, so every 'corrected' row is one observation deep)")


if __name__ == "__main__":
    main()
