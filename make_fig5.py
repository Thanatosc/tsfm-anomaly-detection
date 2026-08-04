"""Figure 5: where the signal is lost, and what recovers it (§5.6).

Two panels, one argument:

  (a) Separation effect size rank-orders detection quality across all eight detectors,
      and the foundation-model residuals sit at the bottom of that axis. Improving the
      forecaster moves a detector along an axis whose whole observed range lies below
      IsolationForest.

  (b) The same score-level moving average applied to every detector leaves the classical
      detectors flat and lifts the foundation models from 25 % to 53 % of the best
      classical detector. The conversion, not the forecast, is where the recoverable
      signal is.

Inputs:  results/mechanism_v2.csv, results/smoothing.csv  (both from analyze_mechanism_v2.py)
Output:  figures/fig5_mechanism.pdf (vector, used by the manuscript) and .png (README)
Run:     .venv/Scripts/python.exe make_fig5.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})

# Vector PDF for the manuscript, PNG for the README. See killtest/figures.py for
# why PDF rather than the EPS Springer nominally prefers.
PNG_DPI = 200

LABEL = {"knn": "k-NN", "ae": "Autoencoder", "lof": "LOF", "iforest": "IsolationForest",
         "timesfm_resid": "TimesFM", "tirex_resid": "TiRex",
         "chronos_base_resid": "Chronos base", "chronos_small_resid": "Chronos small"}
CLASSICAL = ["knn", "ae", "lof", "iforest"]

# Panel (a): the four TSFM residuals sit in a cluster about 0.25 wide on x and 0.015
# tall on y, so offset labels alone do not say which label belongs to which marker.
# They get leader lines to a fanned-out column instead; the classical detectors are
# far enough apart for plain offsets. Positions are in data coordinates.
LEAD_A = {"timesfm_resid": (0.95, 0.128), "tirex_resid": (0.95, 0.100),
          "chronos_base_resid": (0.95, 0.072), "chronos_small_resid": (0.95, 0.044)}
OFF_A = {"knn": (8, 2), "ae": (8, -3), "lof": (8, 2), "iforest": (8, 2)}
OFF_B = {"knn": (5, 4), "ae": (5, -7), "lof": (5, -1), "iforest": (5, 2),
         "timesfm_resid": (5, 7), "tirex_resid": (5, 0),
         "chronos_small_resid": (5, -6), "chronos_base_resid": (5, -17)}


def load_mechanism():
    with open(ROOT / "results" / "mechanism_v2.csv") as f:
        return list(csv.DictReader(f))


def load_smoothing():
    acc = defaultdict(dict)
    with open(ROOT / "results" / "smoothing.csv") as f:
        for r in csv.DictReader(f):
            acc[r["detector"]][int(r["width"])] = float(r["mean_vus"])
    return acc


def main():
    mech, smooth = load_mechanism(), load_smoothing()
    # Stacked rather than side by side: the journal column is about 4.9 in wide, so
    # two panels abreast would be scaled to roughly half size and the axis labels
    # would stop being legible in print. Panels carry only their letter, because
    # Springer asks for no titles inside illustrations; the caption describes them.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 6.8), layout="constrained")

    # ---- (a) separation vs quality -------------------------------------------------
    for r in mech:
        eff, lo, hi = float(r["median_effect"]), float(r["ci_lo"]), float(r["ci_hi"])
        v, is_t = float(r["mean_vus_pr"]), r["family"] == "tsfm"
        col = "tab:red" if is_t else "tab:blue"
        ax1.errorbar(eff, v, xerr=[[eff - lo], [hi - eff]], fmt="none",
                     ecolor=col, elinewidth=0.9, capsize=2.5, alpha=0.75, zorder=2)
        ax1.scatter(eff, v, s=58, marker="o" if is_t else "s", zorder=3, color=col)
        if r["detector"] in LEAD_A:
            ax1.annotate(LABEL[r["detector"]], (eff, v), xytext=LEAD_A[r["detector"]],
                         textcoords="data", fontsize=7.5, color=col,
                         va="center", ha="left", zorder=4,
                         arrowprops=dict(arrowstyle="-", lw=0.6, color=col,
                                         alpha=0.8, shrinkA=2, shrinkB=3))
        else:
            ax1.annotate(LABEL[r["detector"]], (eff, v), textcoords="offset points",
                         xytext=OFF_A[r["detector"]], fontsize=7.5, color=col)
    ax1.set_xlabel("median separation effect size (95 % bootstrap CI)")
    ax1.set_ylabel("mean VUS-PR")
    ax1.set_ylim(0.0, 0.375)
    ax1.set_title("(a)", loc="left", fontweight="bold")
    ax1.legend(handles=[
        plt.Line2D([], [], marker="s", ls="", color="tab:blue", label="classical baseline"),
        plt.Line2D([], [], marker="o", ls="", color="tab:red", label="zero-shot TSFM residual"),
    ], loc="upper left", fontsize=7.5)

    # ---- (b) matched score aggregation ---------------------------------------------
    widths = sorted(next(iter(smooth.values())).keys())
    x = range(len(widths))
    for det, series in smooth.items():
        is_t = det not in CLASSICAL
        ax2.plot(list(x), [series[w] for w in widths],
                 marker="o" if is_t else "s", ms=4.5, lw=1.4,
                 color="tab:red" if is_t else "tab:blue",
                 alpha=0.9 if is_t else 0.55,
                 ls="-" if is_t else "--")
        ax2.annotate(LABEL[det], (len(widths) - 1, series[widths[-1]]),
                     textcoords="offset points", xytext=OFF_B[det], fontsize=7,
                     color="tab:red" if is_t else "tab:blue")

    best_t = [max(v[w] for d, v in smooth.items() if d not in CLASSICAL) for w in widths]
    best_c = [max(v[w] for d, v in smooth.items() if d in CLASSICAL) for w in widths]
    for i, w in enumerate(widths):
        if w in (widths[0], widths[-1]):
            first = i == 0
            ax2.annotate(f"{100 * best_t[i] / best_c[i]:.0f} %", (i, best_t[i]),
                         textcoords="offset points",
                         xytext=(6 if first else -4, 9), fontsize=8,
                         fontweight="bold", ha="left" if first else "right",
                         color="tab:red")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(w) for w in widths])
    ax2.set_xlim(-0.25, len(widths) - 0.25 + 0.9)
    ax2.set_xlabel("moving-average width applied to every detector's scores (points)")
    ax2.set_ylabel("mean VUS-PR")
    ax2.set_title("(b)", loc="left", fontweight="bold")

    fig.savefig(FIG / "fig5_mechanism.pdf")
    fig.savefig(FIG / "fig5_mechanism.png", dpi=PNG_DPI)
    plt.close(fig)
    print("wrote", FIG / "fig5_mechanism.pdf")
    print(f"  panel (b): best TSFM as % of best classical, "
          f"w={widths[0]}: {100 * best_t[0] / best_c[0]:.1f}%  "
          f"w={widths[-1]}: {100 * best_t[-1] / best_c[-1]:.1f}%")


if __name__ == "__main__":
    main()
