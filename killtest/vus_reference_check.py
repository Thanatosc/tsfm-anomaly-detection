"""Validate our VUS-PR implementation against the reference one.

The protocol committed to replacing `killtest.metrics.range_pr_auc` with the reference
VUS-PR implementation and did not (deviation D3). Since VUS-PR carries the paper's
headline, this module measures what that substitution would have cost.

Compares `killtest.metrics.vus_pr` against `vus.metrics.get_metrics(..., metric="vus")`
-- the implementation released by Paparrizos et al. -- on a stratified sample spanning
all four benchmarks, using each series' own buffer width so the comparison is between
formulations rather than between window choices.

    pip install vus            # not a dependency of the harness; install it to run this
    python -m killtest.vus_reference_check

Writes results/vus_reference_check.csv. Needs the persisted score archive (see README).
"""
import os
import csv
import warnings
import collections

import numpy as np

from . import data as D
from .metrics import vus_pr, segments

warnings.filterwarnings("ignore")

SCORE_DIR = os.path.join("results", "scores")
OUT = os.path.join("results", "vus_reference_check.csv")
N_UCR = 60
DETECTORS = ["knn", "lof", "ae", "iforest",
             "timesfm_resid", "tirex_resid", "chronos_base_resid", "chronos_small_resid"]


def targets(n_ucr=N_UCR):
    """Every SMD machine, every NASA channel, PSM, and a deterministic UCR stride."""
    out = []
    files = D.ucr_series_list()
    for i in np.linspace(0, len(files) - 1, n_ucr).round().astype(int):
        p = files[i]
        out.append(("ucr", p.stem, lambda p=p: D.load_ucr(p)))
    for name in D.smd_machines():
        out.append(("smd", name, lambda n=name: D.load_smd(n)))
    for chan, _, _ in D.nasa_channels():
        out.append(("nasa", chan, lambda c=chan: D.load_nasa(c)))
    out.append(("psm", "psm", lambda: D.load_psm(40000)))
    return out


def buffer_width(labels):
    """The ell_max our vus_pr picks for this series: mean anomaly length, clipped."""
    segs = segments(labels)
    if not segs:
        return None
    return int(np.clip(int(np.mean([e - s + 1 for s, e in segs])), 4, 200))


def collect():
    from vus.metrics import get_metrics

    rows = []
    tg = targets()
    print(f"targets: {len(tg)}", flush=True)
    for i, (ds, name, loader) in enumerate(tg):
        try:
            s = loader()
        except Exception as e:
            print(f"  skip {ds}/{name}: {e}", flush=True)
            continue
        lab = np.asarray(s["test_labels"]).astype(int)
        if lab.ndim > 1:
            lab = lab.max(axis=1)
        win = buffer_width(lab)
        if win is None or lab.sum() == 0 or lab.sum() == len(lab):
            continue
        for det in DETECTORS:
            f = os.path.join(SCORE_DIR, f"{ds}__{name}__{det}__default.npz")
            if not os.path.exists(f):
                continue
            sc = np.load(f)["scores"].astype(np.float64)
            if len(sc) != len(lab) or not np.isfinite(sc).all():
                continue
            try:
                ours = float(vus_pr(lab, sc))
                ref = float(get_metrics(sc, lab, metric="vus", slidingWindow=win)["VUS_PR"])
            except Exception:
                continue
            if ours == ours and ref == ref:
                rows.append({"dataset": ds, "series": name, "detector": det,
                             "custom_vus_pr": ours, "reference_vus_pr": ref})
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(tg)}] rows={len(rows)}", flush=True)
    return rows


def report(rows):
    by = collections.defaultdict(lambda: {"c": [], "r": []})
    for x in rows:
        by[x["detector"]]["c"].append(x["custom_vus_pr"])
        by[x["detector"]]["r"].append(x["reference_vus_pr"])

    print(f"\n{'detector':<24}{'n':>5}{'ours':>10}{'reference':>12}{'ratio':>8}")
    means = {}
    for d in DETECTORS:
        if not by[d]["c"]:
            continue
        c, r = float(np.mean(by[d]["c"])), float(np.mean(by[d]["r"]))
        means[d] = (c, r)
        print(f"{d:<24}{len(by[d]['c']):>5}{c:>10.4f}{r:>12.4f}{r / c:>8.2f}")

    base, tsfm = DETECTORS[:4], DETECTORS[4:]
    for label, k in [("ours", 0), ("reference", 1)]:
        bc = max(means[d][k] for d in base if d in means)
        bt = max(means[d][k] for d in tsfm if d in means)
        wc = min(means[d][k] for d in base if d in means)
        print(f"{label:>10}: best classical {bc:.4f} | best TSFM {bt:.4f} | "
              f"{100 * bt / bc:.1f} % | non-overlapping "
              f"{'YES' if wc > bt else 'NO'}")

    a = np.array([x["custom_vus_pr"] for x in rows])
    b = np.array([x["reference_vus_pr"] for x in rows])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    sp = np.corrcoef(rank(a), rank(b))[0, 1]
    pe = np.corrcoef(a, b)[0, 1]
    print(f"\nn={len(a)}  Spearman {sp:.4f}  Pearson {pe:.4f}  "
          f"mean ratio reference/ours {np.mean(b / np.maximum(a, 1e-12)):.3f}")


def main():
    rows = collect()
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "series", "detector",
                                           "custom_vus_pr", "reference_vus_pr"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}: {len(rows)} rows")
    report(rows)


if __name__ == "__main__":
    main()
