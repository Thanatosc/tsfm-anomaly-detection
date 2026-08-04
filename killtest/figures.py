"""Figures for the paper.

No figure carries a title inside the image: Springer asks for that text in the
caption instead, and latex/main.tex holds it there.

Each panel is written twice, as a vector .pdf for the manuscript and a .png for
the repository README. See save() below.

  fig1_pareto      cost-quality Pareto front (quality vs throughput), log-x
  fig2_gap_strata  TSFM-baseline gap by stratum with bootstrap CIs
  fig3_winrate     per-series win rate of TSFM vs best baseline
  fig4_scatter     per-series TSFM vs baseline quality scatter (y=x reference)
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analyze_full import (BASELINES, TSFM_PREFIXES, bootstrap_ci, family, load,
                           per_series_best, strata)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})

# Raster copies are for the repository README only; GitHub will not render a PDF
# inline. 200 dpi is ample for that and keeps the checkout small.
PNG_DPI = 200


# Result rows key metrics by their internal names; figures should carry the form
# the prose uses, so that "VUS-PR" in the text and the axis label are the same
# string to a reader. Unknown keys fall back to the raw name.
METRIC_LABEL = {"vus_pr": "VUS-PR", "auc_pr": "AUC-PR",
                "affil_f1": "affiliation F1", "event_f1": "event F1"}


def metric_label(metric):
    return METRIC_LABEL.get(metric, metric)


def save(fig, stem):
    """Write the vector PDF the manuscript uses, plus a PNG for the README.

    Springer sets a 1200 dpi floor for line art and 600 for combination art. A
    vector PDF clears both at any printed size, which a raster never does once
    the figure is scaled to the text width. EPS is Springer's stated preference
    for vector art, but PostScript has no alpha channel and these panels depend
    on one -- the grid at 0.3, the bars at 0.75, and the scatter at 0.65 where
    the transparency is what shows overlapping points. PDF keeps it; EPS would
    flatten it.
    """
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=PNG_DPI)
    plt.close(fig)


def _warm_throughput():
    """detector-family -> mean warm points/second, from the isolated cost profile.
    Returns {} if the profile has not been produced yet."""
    p = ROOT / "results" / "cost_profile.csv"
    if not p.exists():
        return {}
    import csv
    acc = defaultdict(list)
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok" or not r.get("points_per_s") or r.get("device") != "cuda":
                continue
            acc[r["detector"]].append(float(r["points_per_s"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _cost_key(detector):
    """Map a detector name to its cost-profile key (usage modes share one forward pass)."""
    for p in TSFM_PREFIXES:
        if detector.startswith(p):
            return p
    return detector


def fig_pareto(rows, metric):
    """Quality vs deployment cost. The cost axis is warm throughput (model already
    resident), not per-series wall-clock: charging every series a full model load
    would penalise the foundation models for a one-off cost no deployment pays."""
    warm = _warm_throughput()
    agg = defaultdict(lambda: {"q": [], "t": []})
    for r in rows:
        k = f'{r["detector"]}/{r["tier"]}'
        agg[k]["q"].append(r[metric])
        agg[k]["t"].append(r["runtime_s"] + (r["tune_s"] if r["tune_s"] == r["tune_s"] else 0.0))

    pts = []
    for k, v in agg.items():
        det = k.split("/")[0]
        thr = warm.get(_cost_key(det))
        if thr is None:                      # fall back to measured wall-clock rate
            n = np.nanmean([r["n_test"] for r in rows if r["detector"] == det]) or 1.0
            thr = n / max(float(np.nanmean(v["t"])), 1e-6)
        pts.append((k, float(np.nanmean(v["q"])), thr))

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for k, q, thr in pts:
        is_tsfm = any(k.startswith(p) for p in TSFM_PREFIXES)
        dominated = any(q2 >= q and t2 >= thr and (q2 > q or t2 > thr) for _, q2, t2 in pts)
        ax.scatter(thr, q, s=56 if not dominated else 30,
                   marker="o" if is_tsfm else "s",
                   facecolor=("tab:red" if is_tsfm else "tab:blue") if not dominated else "none",
                   edgecolor="tab:red" if is_tsfm else "tab:blue", zorder=3, linewidths=1.3)
        ax.annotate(k.replace("/default", "").replace("chronos_", "chr-"), (thr, q),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    front = sorted([(t, q) for k, q, t in pts
                    if not any(q2 >= q and t2 >= t and (q2 > q or t2 > t) for _, q2, t2 in pts)])
    ax.plot([p[0] for p in front], [p[1] for p in front], "k--", lw=0.9, alpha=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("throughput, model resident (points/second, log scale) — further right is cheaper")
    ax.set_ylabel(f"mean {metric_label(metric)}")
    h = [plt.Line2D([], [], marker="s", ls="", color="tab:blue", label="classical baseline"),
         plt.Line2D([], [], marker="o", ls="", color="tab:red", label="zero-shot TSFM")]
    ax.legend(handles=h, loc="center left", fontsize=7.5)
    save(fig, "fig1_pareto")


def _gap_by_dim(rows, metric):
    best = per_series_best(rows, metric)
    meta = {(r["dataset"], r["series"]): strata(r) for r in rows}
    dims = defaultdict(lambda: defaultdict(list))
    for key, d in best.items():
        if d["tsfm"] is None or d["baseline"] is None:
            continue
        gap = d["tsfm"][0] - d["baseline"][0]
        for dim, lvl in meta[key].items():
            dims[dim][lvl].append(gap)
    return dims, best


def fig_gap_strata(rows, metric):
    dims, _ = _gap_by_dim(rows, metric)
    order = [d for d in ("dataset", "anomaly_kind", "channels", "length", "contamination") if d in dims]
    labels, means, los, his, ns = [], [], [], [], []
    for dim in order:
        for lvl, gaps in sorted(dims[dim].items(), key=lambda kv: np.mean(kv[1])):
            if len(gaps) < 3:
                continue
            lo, hi = bootstrap_ci(gaps)
            labels.append(f"{dim}: {lvl}"); means.append(np.mean(gaps))
            los.append(lo); his.append(hi); ns.append(len(gaps))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(labels) + 1.2))
    err = np.vstack([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    ax.barh(y, means, xerr=err, color=["tab:red" if m > 0 else "tab:blue" for m in means],
            alpha=0.75, error_kw=dict(lw=0.9, capsize=2.5))
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, ns)], fontsize=7.5)
    ax.set_xlabel(f"mean gap in {metric_label(metric)}   (best TSFM − best baseline)")
    save(fig, "fig2_gap_strata")


def fig_winrate(rows, metric):
    dims, _ = _gap_by_dim(rows, metric)
    order = [d for d in ("dataset", "anomaly_kind", "channels", "length", "contamination") if d in dims]
    labels, wr, ns = [], [], []
    for dim in order:
        for lvl, gaps in sorted(dims[dim].items(), key=lambda kv: np.mean([g > 0 for g in kv[1]])):
            if len(gaps) < 3:
                continue
            labels.append(f"{dim}: {lvl}"); wr.append(float(np.mean([g > 0 for g in gaps]))); ns.append(len(gaps))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(labels) + 1.2))
    ax.barh(y, wr, color="tab:purple", alpha=0.8)
    ax.axvline(0.5, color="k", ls="--", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, ns)], fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of series where a TSFM beats every baseline")
    save(fig, "fig3_winrate")


def fig_scatter(rows, metric):
    best = per_series_best(rows, metric)
    meta = {(r["dataset"], r["series"]): strata(r) for r in rows}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    colors = {"ucr": "tab:blue", "smd": "tab:orange", "nasa": "tab:green", "psm": "tab:red"}
    for key, d in best.items():
        if d["tsfm"] is None or d["baseline"] is None:
            continue
        ax.scatter(d["baseline"][0], d["tsfm"][0], s=14, alpha=0.65,
                   color=colors.get(meta[key]["dataset"], "gray"),
                   label=meta[key]["dataset"])
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"best classical baseline ({metric_label(metric)})")
    ax.set_ylabel(f"best zero-shot TSFM ({metric_label(metric)})")
    hs, ls = ax.get_legend_handles_labels()
    uniq = dict(zip(ls, hs))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7.5, loc="upper left")
    save(fig, "fig4_scatter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "results" / "full_results.csv"))
    ap.add_argument("--metric", default="vus_pr")
    args = ap.parse_args()
    rows = load(Path(args.file))
    print(f"{len(rows)} rows, {len({(r['dataset'], r['series']) for r in rows})} series")
    fig_pareto(rows, args.metric)
    fig_gap_strata(rows, args.metric)
    fig_winrate(rows, args.metric)
    fig_scatter(rows, args.metric)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
