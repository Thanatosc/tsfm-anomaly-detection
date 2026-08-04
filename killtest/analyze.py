"""Pre-registered kill-test decision analysis (criteria fixed in stage1_rq_brief.md BEFORE experiments).

PASS iff:
  (a) exists >=1 stratum where |mean VUS-PR(best TSFM) - mean VUS-PR(best baseline)| >= 0.03, AND
  (b) non-parallel pattern across strata: the TSFM-baseline gap changes sign across strata,
      or gap range (max-min) >= 0.05.
Strata: dataset x anomaly-fraction tercile (computed within dataset), plus UCR domain family.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

CSV = Path(__file__).resolve().parent.parent / "results" / "killtest_results.csv"
TSFMS = {"chronos_bolt_small", "chronos_bolt_base"}
BASELINES = {"iforest", "lof", "ae"}


def load():
    rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "ok" and r["vus_pr"]]
    for r in rows:
        r["vus_pr"] = float(r["vus_pr"]); r["anomaly_frac"] = float(r["anomaly_frac"])
    return rows


def strata_of(rows):
    """Assign each (dataset, series) a stratum label."""
    byds = defaultdict(list)
    for r in rows:
        byds[r["dataset"]].append(r["anomaly_frac"])
    ters = {ds: np.percentile(sorted(set(v)), [33, 66]) for ds, v in byds.items()}
    def lab(r):
        t = ters[r["dataset"]]
        tier = "lowA" if r["anomaly_frac"] <= t[0] else ("midA" if r["anomaly_frac"] <= t[1] else "highA")
        return f'{r["dataset"]}:{tier}'
    return lab


def main():
    rows = load()
    lab = strata_of(rows)
    # per (stratum, series): best TSFM vus, best baseline vus
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        per[lab(r)][r["series"]][r["detector"]] = r["vus_pr"]
    print(f"{'stratum':<14}{'n':>4}{'TSFM':>8}{'base':>8}{'gap':>8}")
    gaps = {}
    for st, series in sorted(per.items()):
        g = []
        for s, d in series.items():
            ts = [v for k, v in d.items() if k in TSFMS]
            bs = [v for k, v in d.items() if k in BASELINES]
            if ts and bs:
                g.append((max(ts), max(bs)))
        if not g:
            continue
        t, b = np.mean([x[0] for x in g]), np.mean([x[1] for x in g])
        gaps[st] = t - b
        print(f"{st:<14}{len(g):>4}{t:>8.3f}{b:>8.3f}{t-b:>+8.3f}")
    gv = np.array(list(gaps.values()))
    a = bool((np.abs(gv) >= 0.03).any())
    b = bool((gv.max() > 0 and gv.min() < 0) or (gv.max() - gv.min() >= 0.05))
    print(f"\n(a) some |gap|>=0.03: {a}   (b) sign flip or range>=0.05: {b}")
    print("KILL TEST:", "PASS" if a and b else "FAIL")
    return a and b


if __name__ == "__main__":
    main()
