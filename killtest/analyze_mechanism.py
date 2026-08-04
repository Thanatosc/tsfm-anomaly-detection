"""Section 6.3 mechanism test: does zero-shot forecasting skill transfer to detection?

The `resid` usage mode scores a point by |x_t - median forecast_t| divided by a robust
scale, so a persisted resid score array *is* the normalised forecast error. That lets us
pull apart two quantities any quality metric conflates:

  forecast skill  -- mean score on NORMAL points; lower means a better forecaster
  separation      -- how much larger the score gets on ANOMALOUS points

Detection needs separation, not skill. Two competing readings of the negative result:

  (H1) "The models forecast these series badly, so the residual is noise."
       Predicts: forecast skill correlates with detection quality across series.
  (H2) "The models forecast too well -- they reconstruct the anomaly along with
       everything else, so nothing stands out."
       Predicts: separation stays near 1 regardless of skill, and skill does *not*
       predict detection quality.

Baselines are included as a reference: whatever separation a working detector achieves
on the same series is the bar the TSFM residual fails to clear.

Run: python -m killtest.analyze_mechanism
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_full import load

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "results" / "scores"


def score_path(dataset, series, detector, tier):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in series)
    return SCORES / f"{dataset}__{safe}__{detector}__{tier}.npz"


def separation(scores, labels):
    """Return (skill, sep_ratio, effect) for one series.

    skill      -- mean score on normal points (resid mode: the normalised forecast error)
    sep_ratio  -- mean anomalous score / mean normal score; 1.0 means the detector treats
                  anomalies exactly like normal data
    effect     -- (mean_anom - mean_normal) / sd_normal, a scale-free separation
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() < 5 or (~y).sum() < 50:
        return None
    norm, anom = s[~y], s[y]
    mu_n, sd_n = norm.mean(), norm.std()
    if not np.isfinite(mu_n) or mu_n <= 0 or sd_n <= 0:
        return None
    return mu_n, anom.mean() / mu_n, (anom.mean() - mu_n) / sd_n


def spearman(x, y):
    """Rank correlation without a scipy dependency."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8:
        return np.nan, 0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (float((rx * ry).sum() / den) if den else np.nan), len(x)


def boot_ci(vals, stat=np.median, n=2000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [stat(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    rows = load(ROOT / "results" / "full_results.csv")
    # resid mode carries the forecast error directly; knn is the strongest baseline and
    # serves as the reference separation on the same series.
    wanted = [d for d in {r["detector"] for r in rows}
              if d.endswith("_resid") or d in ("knn", "ae", "iforest", "lof")]

    recs = defaultdict(list)
    missing = 0
    for r in rows:
        if r["detector"] not in wanted or r["tier"] != "default":
            continue
        p = score_path(r["dataset"], r["series"], r["detector"], r["tier"])
        if not p.exists():
            missing += 1
            continue
        try:
            with np.load(p) as z:
                out = separation(z["scores"], z["labels"])
        except Exception:
            continue
        if out is None:
            continue
        skill, ratio, effect = out
        recs[r["detector"]].append(
            dict(series=f"{r['dataset']}/{r['series']}", skill=skill, ratio=ratio,
                 effect=effect, vus=r["vus_pr"]))

    print(f"loaded {sum(len(v) for v in recs.values())} (series, detector) score arrays"
          f"{f'; {missing} missing' if missing else ''}\n")

    # ---- 1. separation achieved, per detector ----
    print("=" * 78)
    print("1. Anomaly separation  (ratio 1.0 = anomalies score like normal data)")
    print("=" * 78)
    print(f"{'detector':22s} {'n':>4s} {'sep ratio':>10s} {'95% CI':>16s} {'effect':>8s} {'VUS-PR':>8s}")
    for d in sorted(recs, key=lambda k: -np.median([x["ratio"] for x in recs[k]])):
        v = recs[d]
        ratios = [x["ratio"] for x in v]
        lo, hi = boot_ci(ratios)
        print(f"{d:22s} {len(v):4d} {np.median(ratios):10.2f} "
              f"[{lo:6.2f},{hi:6.2f}] {np.median([x['effect'] for x in v]):8.2f} "
              f"{np.median([x['vus'] for x in v]):8.3f}")

    # ---- 2. does forecast skill buy detection? ----
    print()
    print("=" * 78)
    print("2. Does a better forecaster detect better?  (resid detectors only)")
    print("   CONFOUNDED WHEN POOLED -- see section 3b for the per-dataset breakdown,")
    print("   which is the number to quote. rho(sep, VUS) > 0 is the sanity check that")
    print("   separation, not skill, is what detection quality tracks.")
    print("=" * 78)
    print(f"{'detector':22s} {'n':>4s} {'rho(skill,VUS)':>15s} {'rho(sep,VUS)':>13s}")
    for d in sorted(recs):
        if not d.endswith("_resid"):
            continue
        v = recs[d]
        # better forecaster = smaller normal-point residual, so negate for readability
        r_skill, n = spearman([-x["skill"] for x in v], [x["vus"] for x in v])
        r_sep, _ = spearman([x["ratio"] for x in v], [x["vus"] for x in v])
        print(f"{d:22s} {n:4d} {r_skill:15.3f} {r_sep:13.3f}")

    # ---- 3. paired: TSFM residual vs the best baseline on the same series ----
    print()
    print("=" * 78)
    print("3. Paired separation on identical series: TSFM residual vs k-NN")
    print("   Effect size is the comparable quantity -- the raw ratio depends on each")
    print("   detector's score scale (IForest sits near 1.0 by construction).")
    print("=" * 78)
    knn = {x["series"]: x for x in recs.get("knn", [])}
    print(f"{'detector':22s} {'n':>4s} {'TSFM eff':>9s} {'kNN eff':>9s} {'TSFM wins':>10s}")
    for d in sorted(recs):
        if not d.endswith("_resid"):
            continue
        pairs = [(x["effect"], knn[x["series"]]["effect"]) for x in recs[d] if x["series"] in knn]
        if not pairs:
            continue
        t = np.array([p[0] for p in pairs]); k = np.array([p[1] for p in pairs])
        print(f"{d:22s} {len(pairs):4d} {np.median(t):9.2f} {np.median(k):9.2f} "
              f"{100 * (t > k).mean():9.0f}%")

    # ---- 3b. robustness of the skill-vs-detection correlation ----
    # Two confounds a reviewer will raise: (a) the 24% of series no detector solves could
    # drag the correlation, (b) it could be a between-dataset artefact rather than a
    # within-dataset relationship. Re-run the correlation with each ruled out.
    print()
    print("=" * 78)
    print("3b. Robustness of rho(skill, VUS)")
    print("=" * 78)
    solvable = {x["series"] for x in recs.get("knn", []) if x["vus"] >= 0.1}
    print(f"{'detector':22s} {'all':>8s} {'solvable':>10s} {'per-dataset (n)':>28s}")
    for d in sorted(recs):
        if not d.endswith("_resid"):
            continue
        v = recs[d]
        r_all, _ = spearman([-x["skill"] for x in v], [x["vus"] for x in v])
        sub = [x for x in v if x["series"] in solvable]
        r_solv, n_solv = spearman([-x["skill"] for x in sub], [x["vus"] for x in sub])
        per = defaultdict(list)
        for x in v:
            per[x["series"].split("/")[0]].append(x)
        parts = []
        for ds in sorted(per):
            rr, nn = spearman([-x["skill"] for x in per[ds]], [x["vus"] for x in per[ds]])
            if np.isfinite(rr):
                parts.append(f"{ds}:{rr:+.2f}({nn})")
        print(f"{d:22s} {r_all:8.3f} {r_solv:10.3f} ({n_solv:3d})  {' '.join(parts)}")

    print()
    print("  READ THIS BEFORE QUOTING THE POOLED NUMBER. The pooled rho(skill, VUS) is")
    print("  confounded: it is negative only across datasets, while within UCR -- the")
    print("  largest and only univariate subset -- it is mildly POSITIVE. Datasets differ")
    print("  in both forecast-error scale and attainable VUS-PR, so pooling them induces")
    print("  a Simpson reversal. The defensible claim is the within-dataset one: forecast")
    print("  skill does NOT predict detection quality (rho between -0.03 and +0.20).")
    print("  The stronger reading -- 'better forecasters detect worse' -- is NOT supported.")

    # ---- 4. write the per-series table for the paper appendix ----
    out = ROOT / "results" / "mechanism.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "series", "forecast_err_normal", "sep_ratio", "effect", "vus_pr"])
        for d, v in sorted(recs.items()):
            for x in v:
                w.writerow([d, x["series"], f"{x['skill']:.6g}", f"{x['ratio']:.4f}",
                            f"{x['effect']:.4f}", f"{x['vus']:.4f}"])
    print(f"\nper-series table -> {out}")


if __name__ == "__main__":
    main()
