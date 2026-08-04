"""Reproducibility check: are the archived results independent of machine load?

The cost profile had to be discarded because throughput is load-sensitive (deviation D2).
The quality results should not be -- VUS-PR is a deterministic function of the anomaly
scores, which are a deterministic function of data and model. But "should not be" is not
evidence, and the full experiment was collected across many hours under wildly varying
load (1-4 workers, GPU memory thrashing, concurrent analysis jobs).

This re-runs a stratified sample of (series, detector) combinations on the current machine
and compares against the archive, at two levels:

  scores  -- max |new - stored| and Spearman rank correlation (rank is what every
             threshold-swept metric actually consumes; stored scores are float16, so
             exact equality is not expected below ~1e-3 relative)
  metric  -- recomputed VUS-PR vs the value in full_results.csv

If both match, machine load did not touch the reported numbers and no re-run is needed.

Run: python -m killtest.verify_determinism [--n 2] [--detectors knn,ae,...]
"""
import argparse
import random
from pathlib import Path

import numpy as np

from .analyze_full import load
from .analyze_mechanism import score_path, spearman
from .detectors import DETECTORS
from .metrics import vus_pr

ROOT = Path(__file__).resolve().parent.parent
PSM_LEN = 40000  # run_full.py --psm-len default


def load_series(dataset, series):
    from . import data as D
    if dataset == "ucr":
        return D.load_ucr(next(p for p in D.ucr_series_list() if p.stem == series))
    if dataset == "smd":
        return D.load_smd(series)
    if dataset == "nasa":
        return D.load_nasa(series)
    if dataset == "psm":
        # must match run_full.py's --psm-len default; load_psm() unrestricted returns
        # 87 841 points against the 40 000-point prefix the experiment actually used
        return D.load_psm(max_len=PSM_LEN)
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="series per (dataset, detector)")
    ap.add_argument("--detectors", default="knn,iforest,ae,chronos_small_resid,timesfm_quantile,tirex_resid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load(ROOT / "results" / "full_results.csv")
    wanted = args.detectors.split(",")
    rng = random.Random(args.seed)

    # stratified sample: n series per (dataset, detector), default tier only
    buckets = {}
    for r in rows:
        if r["detector"] not in wanted or r["tier"] != "default":
            continue
        buckets.setdefault((r["dataset"], r["detector"]), []).append(r)
    sample = []
    for k in sorted(buckets):
        pool = sorted(buckets[k], key=lambda r: r["series"])
        sample += rng.sample(pool, min(args.n, len(pool)))

    print(f"re-running {len(sample)} (series, detector) combinations on the current machine\n")
    print(f"{'dataset':6s} {'series':22s} {'detector':20s} {'max|dS|':>9s} {'rank rho':>9s} "
          f"{'VUS stored':>11s} {'VUS now':>9s} {'dVUS':>9s}")

    worst_d, worst_rho, worst_vus = 0.0, 1.0, 0.0
    failures = []
    for r in sample:
        item = load_series(r["dataset"], r["series"])
        scores = DETECTORS[r["detector"]](item["train"], item["test"])
        lab = item["test_labels"]

        p = score_path(r["dataset"], r["series"], r["detector"], r["tier"])
        stored = np.load(p)["scores"].astype(np.float64) if p.exists() else None

        v_now = vus_pr(lab, np.asarray(scores, dtype=np.float64))
        v_old = r["vus_pr"]
        dv = abs(v_now - v_old)

        if stored is not None and len(stored) == len(scores):
            new16 = np.asarray(scores, dtype=np.float16).astype(np.float64)
            fin = np.isfinite(new16) & np.isfinite(stored)
            rng_span = np.ptp(stored[fin]) or 1.0
            dmax = float(np.abs(new16[fin] - stored[fin]).max()) / rng_span
            rho, _ = spearman(new16[fin], stored[fin])
        else:
            dmax, rho = float("nan"), float("nan")

        worst_d = max(worst_d, 0.0 if dmax != dmax else dmax)
        worst_rho = min(worst_rho, 1.0 if rho != rho else rho)
        worst_vus = max(worst_vus, dv)
        if dv > 0.005 or (rho == rho and rho < 0.999):
            failures.append((r["dataset"], r["series"], r["detector"], dmax, rho, dv))

        print(f"{r['dataset']:6s} {r['series'][:22]:22s} {r['detector']:20s} "
              f"{dmax:9.2e} {rho:9.5f} {v_old:11.4f} {v_now:9.4f} {dv:9.2e}")

    print()
    print(f"worst normalised score deviation : {worst_d:.2e}")
    print(f"worst score rank correlation     : {worst_rho:.6f}")
    print(f"worst |VUS-PR difference|        : {worst_vus:.2e}")
    print()
    if not failures:
        print("VERDICT: reproduced. Machine load did not affect the reported quality metrics;")
        print("         a full re-run is not warranted.")
    else:
        print(f"VERDICT: {len(failures)} combination(s) did NOT reproduce -- investigate before")
        print("         trusting the archive:")
        for f in failures:
            print(f"  {f[0]}/{f[1]} {f[2]}  dmax={f[3]:.2e} rho={f[4]:.5f} dVUS={f[5]:.2e}")


if __name__ == "__main__":
    main()
