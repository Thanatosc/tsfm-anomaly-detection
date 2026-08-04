"""Mechanism analysis: on which series do zero-shot TSFMs actually win, and why?

Characterises the winning subset against the losing subset on measurable series
properties (length, channels, contamination, anomaly kind, signal predictability),
so the paper can say something causal rather than only "TSFMs lose on average".
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

from .analyze_full import load, per_series_best, strata, bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent


def series_features(dataset, name):
    """Cheap descriptors computed from the raw series (no model involved)."""
    from . import data as D
    try:
        if dataset == "ucr":
            p = next(x for x in D.ucr_series_list() if x.stem == name)
            it = D.load_ucr(p)
        elif dataset == "smd":
            it = D.load_smd(name)
        elif dataset == "nasa":
            it = D.load_nasa(name)
        else:
            it = D.load_psm(max_len=40000)
    except Exception:
        return None
    tr = it["train"]
    x = tr if tr.ndim == 1 else tr[:, int(np.argmax(tr.var(axis=0)))]
    x = np.asarray(x, dtype=float)
    if len(x) < 100:
        return None
    d1 = np.diff(x)
    # lag-1 autocorrelation: how predictable is the signal at all?
    ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if np.std(x) > 0 else 0.0
    # naive-forecast error relative to signal scale: low = easy to forecast
    naive = float(np.mean(np.abs(d1)) / (np.std(x) + 1e-9))
    # spectral concentration: energy share of the dominant non-DC frequency
    f = np.abs(np.fft.rfft(x - x.mean()))
    conc = float(f[1:].max() / (f[1:].sum() + 1e-9)) if len(f) > 2 else 0.0
    return {"ac1": ac1, "naive_err": naive, "spec_conc": conc,
            "cv": float(np.std(x) / (abs(np.mean(x)) + 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "results" / "full_results.csv"))
    ap.add_argument("--metric", default="vus_pr")
    args = ap.parse_args()
    rows = load(Path(args.file))
    best = per_series_best(rows, args.metric)
    meta = {(r["dataset"], r["series"]): strata(r) for r in rows}

    wins, losses = [], []
    detail = []
    for key, d in best.items():
        if d["tsfm"] is None or d["baseline"] is None:
            continue
        gap = d["tsfm"][0] - d["baseline"][0]
        rec = {"key": key, "gap": gap, "tsfm_det": d["tsfm"][1], "base_det": d["baseline"][1],
               "tsfm_val": d["tsfm"][0], "base_val": d["baseline"][0], **meta[key]}
        (wins if gap > 0 else losses).append(rec)
        detail.append(rec)

    print(f"TSFM wins on {len(wins)}/{len(detail)} series ({100*len(wins)/max(1,len(detail)):.1f}%)\n")

    print("=== Which TSFM wins, when one does ===")
    c = defaultdict(int)
    for w in wins:
        c[w["tsfm_det"]] += 1
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {k:26s} {v:3d}")

    print("\n=== Which baseline is the one to beat ===")
    c = defaultdict(int)
    for r in detail:
        c[r["base_det"]] += 1
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {k:26s} {v:3d}")

    print("\n=== Absolute quality on winning vs losing series ===")
    for nm, grp in (("TSFM wins", wins), ("baseline wins", losses)):
        if not grp:
            continue
        t = np.mean([r["tsfm_val"] for r in grp]); b = np.mean([r["base_val"] for r in grp])
        print(f"  {nm:14s} n={len(grp):3d}   TSFM {t:.3f}   baseline {b:.3f}   "
              f"(both low = nobody solves these)")

    print("\n=== Series descriptors: winning vs losing subset ===")
    feats = {}
    for r in detail:
        f = series_features(*r["key"])
        if f:
            feats[r["key"]] = f
    keys = ["ac1", "naive_err", "spec_conc", "cv"]
    print(f"{'feature':12s}{'TSFM wins':>22s}{'baseline wins':>22s}")
    for k in keys:
        w = [feats[r["key"]][k] for r in wins if r["key"] in feats]
        l = [feats[r["key"]][k] for r in losses if r["key"] in feats]
        if not w or not l:
            continue
        wl, wh = bootstrap_ci(w); ll, lh = bootstrap_ci(l)
        print(f"{k:12s}{np.mean(w):>10.3f} [{wl:.2f},{wh:.2f}]{np.mean(l):>10.3f} [{ll:.2f},{lh:.2f}]")

    print("\n=== Hardest series (both families below 0.1) ===")
    hard = [r for r in detail if r["tsfm_val"] < 0.1 and r["base_val"] < 0.1]
    print(f"  {len(hard)}/{len(detail)} series ({100*len(hard)/max(1,len(detail)):.0f}%) are unsolved by every method tested")
    byds = defaultdict(int)
    for r in hard:
        byds[r["dataset"]] += 1
    for k, v in sorted(byds.items()):
        print(f"    {k:6s} {v}")


if __name__ == "__main__":
    main()
