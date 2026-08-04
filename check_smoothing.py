"""Reviewer C3 check: are the baselines getting free score smoothing that the TSFMs do not?

`detectors.windows_to_points` spreads every window score across all w points of the window
and averages, so each classical baseline emits an implicit ~100-point moving average. The
TSFM residual score is raw per-point. This asks what happens to VUS-PR when the TSFM scores
are given the same treatment, using the persisted archive only (no model re-runs).

Run: .venv/Scripts/python.exe check_smoothing.py
"""
import csv
import glob
from collections import defaultdict

import numpy as np

from killtest.analyze_mechanism import score_path
from killtest.metrics import all_metrics

DETECTORS = ["timesfm_resid", "tirex_resid", "chronos_small_resid", "chronos_base_resid", "knn"]
WIDTHS = [1, 32, 100, 300]


def movavg(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


rows = []
for f in glob.glob("results/full_results_*.csv"):
    rows += list(csv.DictReader(open(f)))
seen, ded = set(), []
for r in rows:
    k = (r["dataset"], r["series"], r["detector"], r["tier"])
    if k in seen:
        continue
    seen.add(k)
    ded.append(r)

# series where every detector under test has a fully finite persisted score array
cand = defaultdict(dict)
for r in ded:
    if r["detector"] in DETECTORS and r["tier"] == "default" and r["dataset"] == "ucr":
        cand[(r["dataset"], r["series"])][r["detector"]] = r

usable = []
for key, d in cand.items():
    if len(d) < len(DETECTORS):
        continue
    ok = True
    for det in DETECTORS:
        p = score_path(key[0], key[1], det, "default")
        if not p.exists():
            ok = False
            break
        with np.load(p) as z:
            if not np.isfinite(z["scores"].astype(np.float64)).all():
                ok = False
                break
    if ok:
        usable.append(key)

usable = sorted(usable)[:120]
print(f"UCR series with fully finite arrays for all {len(DETECTORS)} detectors: {len(usable)}\n")

res = {d: {w: [] for w in WIDTHS} for d in DETECTORS}
for key in usable:
    for det in DETECTORS:
        with np.load(score_path(key[0], key[1], det, "default")) as z:
            s = z["scores"].astype(np.float64)
            lab = z["labels"]
        for w in WIDTHS:
            try:
                v = all_metrics(lab, movavg(s, w))["vus_pr"]
            except Exception:
                v = np.nan
            res[det][w].append(v)

print(f"mean VUS-PR over {len(usable)} UCR series, by extra moving-average width applied to the score")
print(f"{'detector':22s}" + "".join(f"{('w=' + str(w)):>10s}" for w in WIDTHS))
for det in DETECTORS:
    line = f"{det:22s}"
    for w in WIDTHS:
        a = np.array(res[det][w], dtype=float)
        line += f"{np.nanmean(a):10.4f}"
    print(line)

print()
knn_raw = np.nanmean(np.array(res["knn"][1], dtype=float))
print(f"k-NN as shipped (w=1, already window-smoothed internally): {knn_raw:.4f}")
for det in DETECTORS:
    if det == "knn":
        continue
    for w in WIDTHS:
        a = np.nanmean(np.array(res[det][w], dtype=float))
        print(f"  {det:22s} w={w:<4d} {a:.4f}   = {100 * a / knn_raw:5.1f}% of k-NN")

print("\nmedians (the mean can be carried by a few series):")
for det in DETECTORS:
    line = f"{det:22s}"
    for w in WIDTHS:
        a = np.array(res[det][w], dtype=float)
        line += f"{np.nanmedian(a):10.4f}"
    print(line)
