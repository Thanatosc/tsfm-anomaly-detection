"""The horizon ramp in the forecast residual, and what removing it is worth.

Scoring uses non-overlapping forecast blocks of horizon 64 (S4.2), so a test point's
residual is taken at whatever horizon its position within the block implies -- one step
ahead at the block start, sixty-four at the end -- and forecast error grows with horizon.
That injects a deterministic period-64 ramp into every foundation-model score which has
nothing to do with whether a point is anomalous, and which the classical detectors, having
no block structure, do not carry.

This matters for reading Table 5: a moving average of width 100 spans more than one block
and would absorb such a ramp whether or not the score conversion were otherwise lossy. So
the ramp has to be removed on its own before the smoothing result can be read as evidence
about the conversion.

    python -m killtest.horizon_check

Reports (1) the within-block profile, with the classical detectors as a control, and
(2) VUS-PR under four conditions: raw, ramp-removed, smoothed, and both. Writes
results/horizon_detrend.csv. Needs the persisted score archive (see README).
"""
import os
import csv
import glob
import collections

import numpy as np

from .metrics import vus_pr

SCORE_DIR = os.path.join("results", "scores")
OUT = os.path.join("results", "horizon_detrend.csv")
HORIZON = 64
TSFM = ["timesfm_resid", "tirex_resid", "chronos_base_resid", "chronos_small_resid"]
CLASSICAL = ["knn", "lof", "ae", "iforest"]
DETECTORS = TSFM + CLASSICAL


def detrend(scores, h=HORIZON):
    """Divide each point by the mean score at its own position mod h.

    Estimated from the series itself, so it uses no labels and no extra inference.
    Anomalies are a small minority of points in this corpus, so the profile is dominated
    by normal behaviour; it removes the ramp and leaves everything else alone.
    """
    n = len(scores)
    k = n // h * h
    if k < 5 * h:
        return scores.copy()
    prof = scores[:k].reshape(-1, h).mean(0)
    if not (prof > 0).any():
        return scores.copy()
    prof = np.where(prof > 0, prof, prof[prof > 0].mean())
    prof = prof / prof.mean()
    return scores / prof[np.arange(n) % h]


def moving_average(scores, w):
    if w <= 1:
        return scores
    return np.convolve(scores, np.ones(w) / w, mode="same")


def series_keys():
    out = []
    for f in sorted(glob.glob(os.path.join(SCORE_DIR, "*__knn__default.npz"))):
        parts = os.path.basename(f)[:-4].split("__")
        out.append((parts[0], parts[1]))
    return out


def load(ds, name, det):
    p = os.path.join(SCORE_DIR, f"{ds}__{name}__{det}__default.npz")
    if not os.path.exists(p):
        return None, None
    with np.load(p) as z:
        s = z["scores"].astype(np.float64)
        y = np.asarray(z["labels"]).astype(int)
    if len(s) != len(y) or not np.isfinite(s).all():
        return None, None
    return s, y


def block_profile(keys, detectors, limit=120):
    """Mean normalised score on NORMAL points by position within the block."""
    acc = {d: np.zeros(HORIZON) for d in detectors}
    cnt = {d: np.zeros(HORIZON) for d in detectors}
    used = collections.Counter()
    for ds, name in keys[:limit]:
        for d in detectors:
            s, y = load(ds, name, d)
            if s is None:
                continue
            m = s[~y.astype(bool)]
            if len(m) < 5 * HORIZON or m.mean() <= 0:
                continue
            m = m / m.mean()
            k = len(m) // HORIZON * HORIZON
            blk = m[:k].reshape(-1, HORIZON)
            acc[d] += blk.sum(0)
            cnt[d] += blk.shape[0]
            used[d] += 1
    return {d: acc[d] / np.maximum(cnt[d], 1) for d in detectors}, used


def control(keys, n_ucr=80):
    ucr = [k for k in keys if k[0] == "ucr"]
    other = [k for k in keys if k[0] != "ucr"]
    idx = np.linspace(0, len(ucr) - 1, min(n_ucr, len(ucr))).round().astype(int)
    targets = [ucr[i] for i in idx] + other
    print(f"series: {len(targets)}", flush=True)

    rows = []
    for i, (ds, name) in enumerate(targets):
        for det in DETECTORS:
            s, y = load(ds, name, det)
            if s is None or y.sum() < 5:
                continue
            try:
                r = {"dataset": ds, "series": name, "detector": det,
                     "raw": vus_pr(y, s),
                     "detrended": vus_pr(y, detrend(s)),
                     "ma100": vus_pr(y, moving_average(s, 100)),
                     "detrended_ma100": vus_pr(y, moving_average(detrend(s), 100))}
            except Exception:
                continue
            if all(v == v for k, v in r.items()
                   if k not in ("dataset", "series", "detector")):
                rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(targets)}] rows={len(rows)}", flush=True)
    return rows


def main():
    keys = series_keys()

    print("1. Mean normalised score on normal points, by quarter of the 64-point block")
    print("   (1.000 = that series' own mean; flat means no block artefact)\n")
    for label, dets in [("foundation models", TSFM), ("classical (control)", CLASSICAL)]:
        prof, used = block_profile(keys, dets)
        print(f"   --- {label} ---")
        print(f"   {'detector':<24}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}"
              f"{'Q4/Q1':>8}{'max/min':>9}{'series':>8}")
        for d in dets:
            p = prof[d]
            q = [p[i * 16:(i + 1) * 16].mean() for i in range(4)]
            print(f"   {d:<24}{q[0]:>8.3f}{q[1]:>8.3f}{q[2]:>8.3f}{q[3]:>8.3f}"
                  f"{q[3] / q[0]:>8.3f}{p.max() / max(p.min(), 1e-9):>9.3f}{used[d]:>8}")
        print()

    print("2. VUS-PR under four conditions\n")
    rows = control(keys)
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "series", "detector", "raw",
                                           "detrended", "ma100", "detrended_ma100"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}: {len(rows)} rows\n")

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for c in ("raw", "detrended", "ma100", "detrended_ma100"):
            by[r["detector"]][c].append(r[c])

    print(f"{'detector':<24}{'raw':>9}{'detrend':>10}{'MA-100':>9}{'both':>9}{'n':>6}")
    means = {}
    for d in DETECTORS:
        if not by[d]["raw"]:
            continue
        v = {c: float(np.mean(by[d][c]))
             for c in ("raw", "detrended", "ma100", "detrended_ma100")}
        means[d] = v
        print(f"{d:<24}{v['raw']:>9.4f}{v['detrended']:>10.4f}"
              f"{v['ma100']:>9.4f}{v['detrended_ma100']:>9.4f}{len(by[d]['raw']):>6}")

    print()
    for c, label in [("raw", "raw"), ("detrended", "ramp removed"),
                     ("ma100", "MA-100"), ("detrended_ma100", "both")]:
        bt = max(means[d][c] for d in TSFM if d in means)
        bc = max(means[d][c] for d in CLASSICAL if d in means)
        print(f"  {label:<14} best TSFM {bt:.4f} / best classical {bc:.4f} "
              f"= {100 * bt / bc:5.1f} %")

    raw = max(means[d]["raw"] for d in TSFM if d in means)
    det = max(means[d]["detrended"] for d in TSFM if d in means)
    ma = max(means[d]["ma100"] for d in TSFM if d in means)
    print(f"\n  gain from removing the ramp alone : {det - raw:+.4f}")
    print(f"  gain from the moving average alone: {ma - raw:+.4f}")
    print(f"  share of the smoothing gain the ramp accounts for: "
          f"{100 * (det - raw) / max(ma - raw, 1e-9):.0f} %")


if __name__ == "__main__":
    main()
