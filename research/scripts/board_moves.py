"""Which KIND of change has ever moved the Kaggle board >= 0.005? (M377)

The board is quantized to 0.001 and nothing under ~0.003 is estimable
(fit_board.py), so the question is asked on DIFFERENCES, not levels: for
every ordered pair of entries, read what differed, and keep the pair only
when the difference is small enough to name.

Population: the 83 `ensemble` rows of comp/board.jsonl that carry
members_named and a numeric board. excluded/member_mean rows are context
only, printed separately at the end.

Pair eligibility (attributable = the pair differs in ONE nameable way):
  - raw member-set symmetric difference 0      -> kernel/fusion/serving
  - stem-level symmetric difference 0          -> recipe/fit/era change
  - raw symmetric difference 1 (count +-1)     -> seat-count change,
    or new-family add if the added stem is new to history
  - raw symmetric difference 2, equal count    -> one-for-one swap
    (same stem: recipe/fit; incoming stem new: new-family; else member swap)
  - raw symmetric difference 2, count +-2      -> two-seat count change
Everything else is multi-change, unattributable. A handful of pairs are
classified from the row's own `what` prose (PROSE_OVERRIDES) where the
mechanical rule would misname a change the entry itself declares.

Stems: trailing _cf/_ff/_s2026/_e12 stripped, plus the aliases below for
names the ledger itself treats as one family (band_dense_long is a longer
band_dense run; resnet_224_short/long are one resnet50d recipe at two epoch
budgets; stackfit is a fit of stack_increments).

"New to history" = the stem appears in no entry with a strictly earlier
`date`. Same-date pairs are ordered by `ref` (the Kaggle submission id,
which increases in submission order), so their deltas carry a real sign;
they are still flagged same_day because sub-day ordering is inferred.

A MOVE is |delta board| >= 0.005. Per class the table reports n_moves,
median and range of the signed delta, n_tries (all attributable pairs in
the class, moves included), hit rate, and where the moves sit against the
0.937-0.944 top band (moves inside the band are sub-resolution per the fit).

Usage: python research/scripts/board_moves.py
"""
import json
from collections import defaultdict
from statistics import median

MOVE = 0.005
BAND_LO = 0.937  # top band: 45 of 120 numeric rows sit in 0.937-0.944

SUFFIXES = ("_s2026", "_e12", "_cf", "_ff")
ALIASES = {
    "band_dense_long": "band_dense",
    "resnet_224_long": "resnet_224",
    "resnet_224_short": "resnet_224",
    "stackfit": "stack_increments",
}

# Pairs the mechanical rule would misname; class taken from the entry's own
# `what` prose. Keyed by frozenset of the two entry ids.
PROSE_OVERRIDES = {
    frozenset({"holdsolo_v1", "v4solo_v1"}): (
        "label_table_swap",
        "identical trunk/serving/count; training label table v2 -> v4b hybrid "
        "(v4solo kernel docstring)",
    ),
    # peak8's members_named are the same eight checkpoints as peak8_v1,
    # recorded without the _s2026 suffix; stem equality would misfile this
    # reship as a recipe change (submit_check bead knee-1sgr: identical set).
    frozenset({"peak8", "peak8_v1"}): (
        "kernel_fusion_serving",
        "reship of peak8_v1's identical eight members, 7 days later; same 0.932",
    ),
}

CLASS_LABELS = {
    "new_family_add": "member added/swapped in from a family no earlier entry held",
    "label_table_swap": "training label table swapped",
    "seat_count_change": "seat count changed (members all previously boarded)",
    "member_swap_known": "one-for-one member swap, both families previously boarded",
    "recipe_fit_change": "same stems, fit/era changed (cf/ff/e12/s2026, twin->shipped, epochs)",
    "kernel_fusion_serving": "identical members; kernel/vote/serving/kit changed",
    "multi_change": "multi-change, unattributable",
}


def load():
    rows = [json.loads(l) for l in open("comp/board.jsonl") if l.strip()]
    data = [r for r in rows if "note" not in r]
    numeric = [r for r in data if isinstance(r.get("board"), (int, float))]
    ens = [
        r for r in numeric
        if r.get("scope") == "ensemble" and r.get("members_named")
    ]
    ctx = [r for r in numeric if r.get("scope") in ("excluded", "member_mean")]
    return sorted(ens, key=lambda r: (r["date"], r["ref"])), ctx


def stem(member):
    for suf in SUFFIXES:
        if member.endswith(suf):
            member = member[: -len(suf)]
            break
    return ALIASES.get(member, member)


def classify(a, b, stems_before):
    """Return (class, detail) for the ordered pair a -> b."""
    ov = PROSE_OVERRIDES.get(frozenset({a["id"], b["id"]}))
    if ov:
        return ov
    sa, sb = set(a["members_named"]), set(b["members_named"])
    ta, tb = {stem(m) for m in sa}, {stem(m) for m in sb}
    only_a, only_b = sorted(sa - sb), sorted(sb - sa)
    d = len(only_a) + len(only_b)
    if d == 0:
        return "kernel_fusion_serving", kernel_detail(a, b)
    if ta == tb:
        return "recipe_fit_change", (
            f"same {len(ta)} stems; suffix/era differs: "
            f"{sorted(ta)}"
        )
    new_in_b = [m for m in only_b if stem(m) not in stems_before]
    if d == 1:
        if only_b:
            if new_in_b:
                return "new_family_add", f"+{only_b[0]} (stem {stem(only_b[0])} new to history)"
            return "seat_count_change", f"{len(sa)}->{len(sb)} seats: +{only_b[0]}"
        return "seat_count_change", f"{len(sa)}->{len(sb)} seats: -{only_a[0]}"
    if d == 2 and len(sa) == len(sb):
        if stem(only_a[0]) == stem(only_b[0]):
            return "recipe_fit_change", f"seat refit: {only_a[0]} -> {only_b[0]}"
        if new_in_b:
            return "new_family_add", (
                f"swap {only_a[0]} -> {only_b[0]} (stem {stem(only_b[0])} new to history)"
            )
        return "member_swap_known", f"swap {only_a[0]} -> {only_b[0]}"
    if d == 2:
        if new_in_b:
            return "new_family_add", (
                f"{len(sa)}->{len(sb)} seats, adds {new_in_b[0]} "
                f"(stem {stem(new_in_b[0])} new to history)"
            )
        return "seat_count_change", f"{len(sa)}->{len(sb)} seats: +{only_b}"
    return "multi_change", f"raw symdiff {d}, stem symdiff {len(ta ^ tb)}"


def kernel_detail(a, b):
    ka, kb = a.get("kernel"), b.get("kernel")
    if ka == kb:
        return f"reship/kit of {ka}"
    return f"kernel {ka} -> {kb}"


def band_zone(lo, hi):
    if lo >= BAND_LO:
        return "within band (both >=0.937)"
    if hi < 0.930:
        return "below 0.930"
    return "crosses into band" if hi >= BAND_LO else "mid (0.930-0.936)"


def main():
    ens, ctx = load()
    print(f"== population: {len(ens)} ensemble rows with members_named "
          f"(numeric board) ==")
    print(f"board range {min(r['board'] for r in ens):.3f}-"
          f"{max(r['board'] for r in ens):.3f}; move threshold {MOVE:.3f}\n")

    # stems present at strictly earlier dates, for the new-to-history test
    first_date = {}
    for r in ens:
        for m in r["members_named"]:
            s = stem(m)
            if s not in first_date or r["date"] < first_date[s]:
                first_date[s] = r["date"]

    pairs = []
    for i, a in enumerate(ens):
        for b in ens[i + 1:]:
            same_day = a["date"] == b["date"]
            stems_before = {
                s for s, d0 in first_date.items() if d0 < a["date"]
            }
            cls, detail = classify(a, b, stems_before)
            delta = round(b["board"] - a["board"], 3)
            pairs.append({
                "a": a["id"], "b": b["id"],
                "da": a["date"], "db": b["date"],
                "ba": a["board"], "bb": b["board"],
                "delta": delta, "cls": cls, "detail": detail,
                "na": len(a["members_named"]), "nb": len(b["members_named"]),
                "same_day": same_day,
                "move": abs(delta) >= MOVE - 1e-9,
            })

    n_moves = sum(p["move"] for p in pairs)
    print(f"ordered pairs: {len(pairs)}; moves |delta|>={MOVE:.3f}: {n_moves}\n")

    by_cls = defaultdict(list)
    for p in pairs:
        by_cls[p["cls"]].append(p)

    order = [c for c in CLASS_LABELS if c in by_cls]
    rows = []
    for cls in order:
        ps = by_cls[cls]
        mv = [p for p in ps if p["move"]]
        deltas = [p["delta"] for p in mv]
        rows.append((cls, ps, mv, deltas))

    hdr = (f"{'class':<24}{'n_moves':>8}{'up/dn':>7}{'median':>8}{'range':>16}"
           f"{'n_tries':>8}{'hit_rate':>9}  band placement of moves")
    print(hdr)
    print("-" * len(hdr))
    for cls, ps, mv, deltas in rows:
        if cls == "multi_change":
            continue
        if deltas:
            med = f"{median(deltas):+.3f}"
            rng = f"{min(deltas):+.3f}..{max(deltas):+.3f}"
            zones = "; ".join(sorted({
                band_zone(min(p["ba"], p["bb"]), max(p["ba"], p["bb"]))
                for p in mv
            }))
        else:
            med, rng, zones = "-", "-", "-"
        up = sum(1 for p in mv if p["delta"] > 0)
        rate = f"{len(mv)}/{len(ps)}"
        print(f"{cls:<24}{len(mv):>8}{f'{up}/{len(mv)-up}':>7}{med:>8}"
              f"{rng:>16}{len(ps):>8}{rate:>9}  {zones}")
        solo_mv = [p for p in mv if p["na"] == 1 and p["nb"] == 1]
        solo_ps = [p for p in ps if p["na"] == 1 and p["nb"] == 1]
        multi_mv = len(mv) - len(solo_mv)
        multi_ps = len(ps) - len(solo_ps)
        flat_max = max((abs(p["delta"]) for p in ps if not p["move"]), default=None)
        fm = f"{flat_max:.3f}" if flat_max is not None else "-"
        print(f"  {'':<22}solo-vs-solo (both n=1): {len(solo_mv)}/{len(solo_ps)}; "
              f"pairs involving an ensemble: {multi_mv}/{multi_ps}; "
              f"max |delta| among flat tries: {fm}")
    mc = by_cls.get("multi_change", [])
    mc_mv = [p for p in mc if p["move"]]
    print(f"{'multi_change':<24}{len(mc_mv):>8}{'':>8}{'':>16}{len(mc):>8}"
          f"{'-':>9}  (unattributable by design)")

    print("\n== classes ranked by hit rate (moves / attributable tries) ==")
    ranked = sorted(
        ((c, len(m), len(ps)) for c, ps, m, _ in rows if c != "multi_change"),
        key=lambda t: -(t[1] / t[2] if t[2] else 0),
    )
    for cls, nm, nt in ranked:
        print(f"  {cls:<24}{nm:>3}/{nt:<3} = {nm / nt if nt else 0:.2f}   "
              f"{CLASS_LABELS[cls]}")

    print("\n== example pairs per class (largest |delta| moves first, then "
          "largest flat tries) ==")
    for cls, ps, mv, _ in rows:
        if cls == "multi_change":
            continue
        print(f"\n-- {cls}: {CLASS_LABELS[cls]}")
        show = sorted(ps, key=lambda p: (not p["move"], -abs(p["delta"])))[:3]
        for p in show:
            sd = " [same-day, ordered by submission ref]" if p["same_day"] else ""
            tag = "MOVE" if p["move"] else "flat"
            print(f"  {tag} {p['a']} ({p['da']} {p['ba']:.3f}) -> "
                  f"{p['b']} ({p['db']} {p['bb']:.3f})  d={p['delta']:+.3f}{sd}")
            print(f"       {p['detail']}")

    print("\n== every move, per class (full list) ==")
    for cls, ps, mv, _ in rows:
        if cls == "multi_change" or not mv:
            continue
        up = sum(1 for p in mv if p["delta"] > 0)
        print(f"\n-- {cls}: {len(mv)} moves ({up} up, {len(mv) - up} down)")
        for p in sorted(mv, key=lambda p: p["delta"]):
            sd = " [same-day]" if p["same_day"] else ""
            print(f"  {p['delta']:+.3f}  {p['a']} ({p['da']} {p['ba']:.3f}) -> "
                  f"{p['b']} ({p['db']} {p['bb']:.3f}){sd}  {p['detail']}")

    print(f"\n== unattributable multi-change pairs: {len(mc_mv)} of {n_moves} "
          f"moves (of {len(mc)} multi-change pairs total) ==")
    print("largest |delta| multi-change moves, for the record:")
    for p in sorted(mc_mv, key=lambda p: -abs(p["delta"]))[:8]:
        sd = " [same-day]" if p["same_day"] else ""
        print(f"  {p['a']} ({p['da']} {p['ba']:.3f}) -> {p['b']} "
              f"({p['db']} {p['bb']:.3f})  d={p['delta']:+.3f}{sd}  {p['detail']}")

    print("\n== context only (excluded / member_mean rows; NOT part of the "
          "83-row population) ==")
    for r in sorted(ctx, key=lambda r: r.get("date") or ""):
        what = (r.get("what") or r.get("why") or "").replace("\n", " ")
        print(f"  {r.get('scope'):<12}{r.get('date', '?'):<12}{r['board']:.3f} "
              f"{r.get('id', '(no id)'):<16}{what[:110]}")


if __name__ == "__main__":
    main()
