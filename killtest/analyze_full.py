"""Break-even and Pareto analysis for the full experiment.

Produces:
  1. Headline table: mean metric by detector family, per dataset.
  2. Stratified break-even: TSFM-vs-baseline gap by anomaly kind, channel count,
     series length, anomaly fraction.
  3. Cost-quality Pareto front (quality vs wall-clock), flagging dominated methods.
  4. Decision map: for each stratum, the recommended method and the cost multiplier
     of using the best TSFM instead.
"""
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

TSFM_PREFIXES = ("chronos_small", "chronos_base", "timesfm", "tirex")
BASELINES = ("iforest", "lof", "knn", "ae")
METRICS = ("vus_pr", "auc_pr", "affil_f1", "event_f1")


def family(det):
    return "tsfm" if any(det.startswith(p) for p in TSFM_PREFIXES) else "baseline"


def load(path):
    """Load one CSV, or every shard file if `path` names the base results file."""
    paths = [path]
    if path.name == "full_results.csv":
        paths = sorted(path.parent.glob("full_results*.csv"))
    rows, seen = [], set()
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["status"] != "ok" or not r["vus_pr"]:
                    continue
                key = (r["dataset"], r["series"], r["detector"], r["tier"])
                if key in seen:
                    continue
                seen.add(key)
                for k in METRICS + ("runtime_s", "tune_s", "peak_mb", "anomaly_frac"):
                    r[k] = float(r[k]) if r[k] not in ("", None) else np.nan
                for k in ("n_test", "n_channels"):
                    r[k] = int(r[k])
                r["family"] = family(r["detector"])
                rows.append(r)
    return rows


def strata(r):
    """Return dict of stratum-name -> level for one row."""
    n = r["n_test"]
    length = "short" if n < 10000 else ("medium" if n < 50000 else "long")
    chan = "univariate" if r["n_channels"] == 1 else ("low_dim" if r["n_channels"] <= 10 else "high_dim")
    af = r["anomaly_frac"]
    contam = "rare" if af < 0.01 else ("moderate" if af < 0.05 else "frequent")
    out = {"dataset": r["dataset"], "length": length, "channels": chan, "contamination": contam}
    if r["anomaly_kind"]:
        out["anomaly_kind"] = r["anomaly_kind"]
    return out


def per_series_best(rows, metric, tier_filter=None):
    """(dataset, series) -> {'tsfm': (best_val, det, runtime), 'baseline': (...)}"""
    acc = defaultdict(lambda: {"tsfm": None, "baseline": None})
    for r in rows:
        if tier_filter and r["tier"] not in tier_filter:
            continue
        v = r[metric]
        if v != v:
            continue
        key = (r["dataset"], r["series"])
        cur = acc[key][r["family"]]
        if cur is None or v > cur[0]:
            acc[key][r["family"]] = (v, r["detector"], r["runtime_s"], r["tier"])
    return acc


def bootstrap_ci(vals, n=2000, seed=0):
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    a = np.asarray(vals)
    means = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def headline(rows, metric):
    print(f"\n=== Headline: mean {metric} by detector (tier=default unless noted) ===")
    by = defaultdict(list)
    for r in rows:
        by[(r["detector"], r["tier"])].append(r[metric])
    print(f"{'detector':26s}{'tier':9s}{'n':>5}{'mean':>8}{'median':>8}{'  95% CI'}")
    for (det, tier), v in sorted(by.items(), key=lambda kv: -np.nanmean(kv[1])):
        v = [x for x in v if x == x]
        lo, hi = bootstrap_ci(v)
        print(f"{det:26s}{tier:9s}{len(v):>5}{np.mean(v):>8.3f}{np.median(v):>8.3f}  [{lo:.3f},{hi:.3f}]")


def stratified_gap(rows, metric, tiers_baseline=("default", "tuned")):
    print(f"\n=== Stratified break-even ({metric}); gap = best TSFM - best baseline ===")
    best = per_series_best(rows, metric)
    meta = {(r["dataset"], r["series"]): strata(r) for r in rows}
    dims = defaultdict(lambda: defaultdict(list))
    for key, d in best.items():
        if d["tsfm"] is None or d["baseline"] is None:
            continue
        gap = d["tsfm"][0] - d["baseline"][0]
        for dim, lvl in meta[key].items():
            dims[dim][lvl].append(gap)
    for dim, levels in dims.items():
        print(f"\n-- {dim} --")
        print(f"{'level':16s}{'n':>5}{'mean gap':>10}{'  95% CI':>18}{'  TSFM wins':>12}")
        for lvl, gaps in sorted(levels.items(), key=lambda kv: -np.mean(kv[1])):
            lo, hi = bootstrap_ci(gaps)
            wins = float(np.mean([g > 0 for g in gaps]))
            print(f"{str(lvl):16s}{len(gaps):>5}{np.mean(gaps):>+10.3f}   [{lo:+.3f},{hi:+.3f}]{wins:>11.0%}")
    return dims


def pareto(rows, metric):
    print(f"\n=== Cost-quality Pareto ({metric} vs mean wall-clock seconds) ===")
    agg = defaultdict(lambda: {"q": [], "t": [], "tune": [], "mem": []})
    for r in rows:
        k = f'{r["detector"]}/{r["tier"]}'
        agg[k]["q"].append(r[metric])
        agg[k]["t"].append(r["runtime_s"] + (r["tune_s"] if r["tune_s"] == r["tune_s"] else 0.0))
        agg[k]["mem"].append(r["peak_mb"] if r["peak_mb"] == r["peak_mb"] else 0.0)
    pts = [(k, float(np.nanmean(v["q"])), float(np.nanmean(v["t"])), float(np.nanmean(v["mem"])))
           for k, v in agg.items()]
    front = []
    for k, q, t, m in pts:
        dominated = any(q2 >= q and t2 <= t and (q2 > q or t2 < t) for _, q2, t2, _ in pts)
        front.append((k, q, t, m, not dominated))
    print(f"{'method/tier':32s}{metric:>9}{'sec':>9}{'peakMB':>9}  frontier")
    for k, q, t, m, on in sorted(front, key=lambda x: -x[1]):
        print(f"{k:32s}{q:>9.3f}{t:>9.2f}{m:>9.0f}  {'YES' if on else ''}")
    return front


def decision_map(rows, metric):
    print(f"\n=== Decision map ({metric}): per stratum, what to deploy ===")
    best = per_series_best(rows, metric)
    meta = {(r["dataset"], r["series"]): strata(r) for r in rows}
    cells = defaultdict(list)
    for key, d in best.items():
        if d["tsfm"] is None or d["baseline"] is None:
            continue
        m = meta[key]
        cell = (m["dataset"], m.get("anomaly_kind", "-"), m["channels"])
        cells[cell].append((d["tsfm"], d["baseline"]))
    print(f"{'dataset':8s}{'kind':12s}{'channels':12s}{'n':>4}{'TSFM':>8}{'base':>8}{'gap':>8}{'cost x':>9}  verdict")
    for cell, vals in sorted(cells.items()):
        t = np.mean([v[0][0] for v in vals]); b = np.mean([v[1][0] for v in vals])
        tt = np.mean([v[0][2] for v in vals]); bt = np.mean([max(v[1][2], 1e-3) for v in vals])
        gap = t - b
        verdict = "TSFM" if gap > 0.02 else ("baseline" if gap < -0.02 else "tie")
        print(f"{cell[0]:8s}{str(cell[1]):12s}{cell[2]:12s}{len(vals):>4}{t:>8.3f}{b:>8.3f}"
              f"{gap:>+8.3f}{tt / bt:>9.1f}  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RES / "full_results.csv"))
    ap.add_argument("--metric", default="vus_pr")
    args = ap.parse_args()
    rows = load(Path(args.file))
    print(f"loaded {len(rows)} rows | "
          f"{len({(r['dataset'], r['series']) for r in rows})} series | "
          f"{len({r['detector'] for r in rows})} detectors")
    headline(rows, args.metric)
    stratified_gap(rows, args.metric)
    pareto(rows, args.metric)
    decision_map(rows, args.metric)


if __name__ == "__main__":
    main()
