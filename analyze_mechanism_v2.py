"""Post-repair re-derivation of the mechanism analysis and the score-aggregation control.

Replaces `analyze_mechanism.py` for the revised §5.6/§6.3. Two changes force the rewrite:

  F3  The published monotone ordering was computed on per-detector subsets whose membership
      was decided by float16 saturation, not by anything meaningful. Everything here runs on
      a single matched series set where every detector has a fully finite array, and every
      effect size carries a bootstrap CI.

  F4  `detectors.windows_to_points` spreads each window score over all w points, so the
      classical baselines emit an implicit ~100-point moving average while the TSFM residual
      is raw per-point. The `smooth` mode measures what happens when that asymmetry is
      removed, which is a label-free, inference-free post-processing step.

Also runs the within-series test the original analysis missed: across the four backends on
the *same* series, does the one that forecasts best also detect best? That removes series
difficulty entirely and is the cleanest available test of "forecast skill buys detection".

Run: .venv/Scripts/python.exe analyze_mechanism_v2.py [mechanism|smooth|both]
"""
import csv
import glob
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from killtest.analyze_mechanism import score_path, separation
from killtest.metrics import all_metrics

TSFM_RESID = ["timesfm_resid", "tirex_resid", "chronos_base_resid", "chronos_small_resid"]
CLASSICAL = ["knn", "ae", "lof", "iforest"]
ALL8 = CLASSICAL + TSFM_RESID
WIDTHS = [1, 32, 100, 300]
RNG = np.random.default_rng(0)


def boot_ci(v, n=2000, stat=np.median):
    v = np.asarray(v, dtype=float)
    if len(v) < 3:
        return (np.nan, np.nan)
    draws = RNG.choice(v, (n, len(v)), replace=True)
    s = stat(draws, axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load_rows():
    rows, seen = [], set()
    for f in sorted(glob.glob("results/full_results_*.csv")):
        for r in csv.DictReader(open(f)):
            k = (r["dataset"], r["series"], r["detector"], r["tier"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def matched_series(rows, detectors):
    have = defaultdict(dict)
    for r in rows:
        if r["detector"] in detectors and r["tier"] == "default":
            have[(r["dataset"], r["series"])][r["detector"]] = r
    out = []
    for key, d in have.items():
        if len(d) < len(detectors):
            continue
        ok = True
        for det in detectors:
            p = score_path(key[0], key[1], det, "default")
            if not p.exists():
                ok = False
                break
            with np.load(p) as z:
                if not np.isfinite(z["scores"].astype(np.float64)).all():
                    ok = False
                    break
        if ok:
            out.append(key)
    return sorted(out), have


def movavg(x, w):
    return x if w <= 1 else np.convolve(x, np.ones(w) / w, mode="same")


def run_mechanism(rows):
    keys, have = matched_series(rows, ALL8)
    ds_count = defaultdict(int)
    for d, _ in keys:
        ds_count[d] += 1
    print(f"matched series (all 8 detectors fully finite): {len(keys)}  {dict(ds_count)}\n")

    eff, vus, skill = defaultdict(list), defaultdict(list), defaultdict(list)
    per_series = defaultdict(dict)
    for key in keys:
        for det in ALL8:
            with np.load(score_path(key[0], key[1], det, "default")) as z:
                s = z["scores"].astype(np.float64)
                lab = z["labels"]
            out = separation(s, lab)
            if out is None:
                continue
            sk, _, e = out
            v = float(have[key][det]["vus_pr"])
            eff[det].append(e)
            vus[det].append(v)
            skill[det].append(sk)
            per_series[key][det] = (sk, e, v)

    print("=" * 96)
    print("1. Separation vs detection quality, matched series, bootstrap CIs on the effect size")
    print("=" * 96)
    print(f"{'detector':24s} {'n':>4s} {'median effect':>14s} {'95% CI':>20s} {'mean VUS-PR':>12s}")
    order = sorted(ALL8, key=lambda d: -np.median(eff[d]))
    seq = []
    summary = []
    for d in order:
        lo, hi = boot_ci(eff[d])
        m, v = np.median(eff[d]), np.mean(vus[d])
        seq.append(v)
        summary.append([d, len(eff[d]), m, lo, hi, v])
        print(f"{d:24s} {len(eff[d]):4d} {m:14.3f} {f'[{lo:.3f}, {hi:.3f}]':>20s} {v:12.4f}")
    inv = [(order[i], order[i + 1]) for i in range(len(seq) - 1) if seq[i] < seq[i + 1]]
    print(f"\nmonotone in VUS-PR along the effect ordering: {not inv}")
    if inv:
        for a, b in inv:
            print(f"  INVERSION: {a} (effect {np.median(eff[a]):.3f}, VUS {np.mean(vus[a]):.4f})"
                  f" ranks above {b} (effect {np.median(eff[b]):.3f}, VUS {np.mean(vus[b]):.4f})")

    print("\n" + "=" * 96)
    print("2. Paired against k-NN on the same series")
    print("=" * 96)
    print(f"{'detector':24s} {'median effect':>14s} {'k-NN effect':>12s} {'ratio':>8s} {'beats k-NN':>11s}")
    paired = {}
    for d in order:
        if d == "knn":
            continue
        pairs = [(per_series[k][d][1], per_series[k]["knn"][1])
                 for k in keys if d in per_series[k] and "knn" in per_series[k]]
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        ratio = np.median(b) / max(np.median(a), 1e-9)
        beats = 100 * np.mean(a > b)
        paired[d] = (ratio, beats)
        print(f"{d:24s} {np.median(a):14.3f} {np.median(b):12.3f} "
              f"{ratio:8.1f} {beats:10.1f}%")

    with open("results/mechanism_v2.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["detector", "family", "n_series", "median_effect", "ci_lo", "ci_hi",
                     "mean_vus_pr", "knn_over_detector", "beats_knn_pct"])
        for d, n, m, lo, hi, v in summary:
            r, bk = paired.get(d, ("", ""))
            wr.writerow([d, "tsfm" if d in TSFM_RESID else "classical", n,
                         f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{v:.4f}",
                         f"{r:.2f}" if r != "" else "", f"{bk:.1f}" if bk != "" else ""])
    print(f"\nwrote results/mechanism_v2.csv ({len(summary)} rows, {len(keys)} matched series)")

    print("\n" + "=" * 96)
    print("3. Does forecast skill predict detection quality? (rho of forecast error vs VUS-PR)")
    print("     positive rho = larger forecast ERROR goes with HIGHER VUS-PR")
    print("=" * 96)
    print(f"{'detector':24s} {'pooled':>10s}   per dataset")
    corr_rows = []
    for d in TSFM_RESID:
        pooled = spearmanr(skill[d], vus[d]).statistic
        cells, per_ds = [], {}
        for ds in ("ucr", "nasa", "smd", "psm"):
            idx = [i for i, k in enumerate(keys) if k[0] == ds and d in per_series[k]]
            if len(idx) < 10:
                cells.append(f"{ds}:n={len(idx)}")
                per_ds[ds] = ("", len(idx))
                continue
            sk = [per_series[keys[i]][d][0] for i in idx]
            vv = [per_series[keys[i]][d][2] for i in idx]
            rho_ds = spearmanr(sk, vv).statistic
            per_ds[ds] = (rho_ds, len(idx))
            cells.append(f"{ds}:{rho_ds:+.2f}(n={len(idx)})")
        corr_rows.append((d, pooled, len(skill[d]), per_ds))
        print(f"{d:24s} {pooled:+10.2f}   {'  '.join(cells)}")

    print("\n" + "=" * 96)
    print("4. WITHIN-SERIES test: across the 4 backends on the SAME series, does the best")
    print("   forecaster also detect best? (removes series difficulty entirely)")
    print("=" * 96)
    rhos = []
    for key in keys:
        sk = [per_series[key][d][0] for d in TSFM_RESID if d in per_series[key]]
        vv = [per_series[key][d][2] for d in TSFM_RESID if d in per_series[key]]
        if len(sk) < 4 or len(set(vv)) < 2:
            continue
        r = spearmanr(sk, vv).statistic
        if r == r:
            rhos.append(r)
    rhos = np.array(rhos)
    lo, hi = boot_ci(rhos, stat=np.mean)
    print(f"n = {len(rhos)} series with all four backends")
    print(f"mean within-series rho(forecast error, VUS-PR) = {rhos.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"fraction of series with rho > 0: {100 * np.mean(rhos > 0):.1f}%")

    with open("results/correlation_levels.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["level", "backend", "rho", "ci_lo", "ci_hi", "n"])
        for d, pooled, n_pooled, per_ds in corr_rows:
            wr.writerow(["pooled", d, f"{pooled:+.4f}", "", "", n_pooled])
            for ds, (rho_ds, n_ds) in per_ds.items():
                wr.writerow([f"within_{ds}", d,
                             f"{rho_ds:+.4f}" if rho_ds != "" else "", "", "", n_ds])
        wr.writerow(["within_series", "all_four_backends", f"{rhos.mean():+.4f}",
                     f"{lo:+.4f}", f"{hi:+.4f}", len(rhos)])
    print(f"wrote results/correlation_levels.csv  "
          f"(within-series rho {rhos.mean():+.3f} on n={len(rhos)})")
    print("\ninterpretation: a negative mean rho means that, on a given series, the backend that")
    print("forecasts it best also tends to detect on it best, which is the opposite of the")
    print("mechanism proposed by the prior study (over-accurate forecasting hiding the anomaly).")


def run_smooth(rows):
    keys, have = matched_series(rows, ALL8)
    print(f"matched series: {len(keys)}\n")
    res = {d: {w: [] for w in WIDTHS} for d in ALL8}
    for i, key in enumerate(keys, 1):
        for det in ALL8:
            with np.load(score_path(key[0], key[1], det, "default")) as z:
                s = z["scores"].astype(np.float64)
                lab = z["labels"]
            for w in WIDTHS:
                try:
                    res[det][w].append(all_metrics(lab, movavg(s, w))["vus_pr"])
                except Exception:
                    res[det][w].append(np.nan)
        if i % 40 == 0:
            print(f"  {i}/{len(keys)}", flush=True)

    with open("results/smoothing.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["detector", "width", "mean_vus", "median_vus", "n"])
        for d in ALL8:
            for w in WIDTHS:
                a = np.array(res[d][w], dtype=float)
                wr.writerow([d, w, f"{np.nanmean(a):.4f}", f"{np.nanmedian(a):.4f}", len(a)])

    print("\n" + "=" * 96)
    print("Score-level moving average applied to every detector equally (mean VUS-PR)")
    print("=" * 96)
    print(f"{'detector':24s}" + "".join(f"{('w=' + str(w)):>11s}" for w in WIDTHS))
    for d in ALL8:
        print(f"{d:24s}" + "".join(f"{np.nanmean(np.array(res[d][w], dtype=float)):11.4f}"
                                   for w in WIDTHS))
    print(f"\n{'(medians)':24s}" + "".join(f"{('w=' + str(w)):>11s}" for w in WIDTHS))
    for d in ALL8:
        print(f"{d:24s}" + "".join(f"{np.nanmedian(np.array(res[d][w], dtype=float)):11.4f}"
                                   for w in WIDTHS))

    print("\nbest TSFM vs best classical at MATCHED post-processing:")
    for w in WIDTHS:
        bt = max(np.nanmean(np.array(res[d][w], dtype=float)) for d in TSFM_RESID)
        bc = max(np.nanmean(np.array(res[d][w], dtype=float)) for d in CLASSICAL)
        print(f"  w={w:<4d} best TSFM {bt:.4f}   best classical {bc:.4f}   "
              f"TSFM reaches {100 * bt / bc:.1f}%")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    rows = load_rows()
    if mode in ("mechanism", "both"):
        run_mechanism(rows)
    if mode in ("smooth", "both"):
        print("\n\n")
        run_smooth(rows)
