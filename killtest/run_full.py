"""Full experiment runner.

Design decisions (pre-registered in stage1_rq_brief.md / killtest_report.md):
  - Baselines are reported in three tiers: default / tuned (budgeted random search on a
    held-out split of the TRAIN region only) / oracle (best test config, upper bound only).
    Tuning cost (wall-clock) is recorded so it enters the break-even accounting.
  - TSFMs are zero-shot; their "tuning cost" is zero by construction.
  - Cost metrics: wall-clock, peak GPU memory. CPU-only timings via --device cpu.
"""
import argparse, csv, gc, os, time, traceback
from pathlib import Path
import numpy as np
import torch

from . import data as D
from .detectors import (BASELINE_FNS, BASELINE_GRIDS, DETECTORS, TSFM_BACKENDS,
                        run_detector, tsfm_scores_all_modes)
from .metrics import all_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SCORES = OUT / "scores"
SCORES.mkdir(exist_ok=True)
FIELDS = ["dataset", "series", "domain", "anomaly_kind", "detector", "tier", "config",
          "auc_pr", "vus_pr", "affil_f1", "event_f1", "runtime_s", "tune_s", "peak_mb",
          "n_test", "n_channels", "anomaly_frac", "status"]


def save_scores(dataset, series, detector, tier, labels, scores):
    """Persist anomaly scores so any future metric can be recomputed without
    re-running the detectors.

    float32, not float16: float16 saturates at 65504, and k-NN/AE distance scores on
    NASA and SMD routinely exceed that. The original float16 archive silently stored inf
    in 1213 of 5504 arrays (22%), which corrupted every analysis recomputed from it while
    leaving full_results.csv correct, since the metrics there were computed before saving.
    float32 costs ~420 MB for the full archive.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in series)
    p = SCORES / f"{dataset}__{safe}__{detector}__{tier}.npz"
    try:
        np.savez_compressed(p, scores=np.asarray(scores, dtype=np.float32),
                            labels=np.asarray(labels, dtype=np.int8))
    except Exception as e:
        print(f"  [warn] could not save scores for {p.name}: {e}", flush=True)


def csv_path(tag):
    return OUT / f"full_results{('_' + tag) if tag else ''}.csv"


def done_keys(path):
    """Union of completed keys across ALL shard files, so a worker never redoes work
    another worker (or an earlier single-process run) already finished."""
    keys = set()
    for p in sorted(OUT.glob("full_results*.csv")):
        try:
            with open(p) as f:
                keys |= {(r["dataset"], r["series"], r["detector"], r["tier"])
                         for r in csv.DictReader(f)}
        except Exception:
            continue
    return keys


def append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def base_row(dataset, item):
    lab = item["test_labels"]
    return dict(dataset=dataset, series=item["name"], domain=item["domain"],
                anomaly_kind=item.get("anomaly_kind", ""), n_test=len(lab),
                n_channels=1 if item["test"].ndim == 1 else item["test"].shape[1],
                anomaly_frac=round(float(np.mean(lab)), 5),
                tier="", config="", auc_pr="", vus_pr="", affil_f1="", event_f1="",
                runtime_s="", tune_s="", peak_mb="", status="ok")


def fill(row, m, rt, peak, tune_s=""):
    row.update({k: (round(v, 5) if v == v else "") for k, v in m.items()})
    row.update(runtime_s=round(rt, 2), peak_mb=round(peak, 1), tune_s=tune_s)
    return row


def tune_baseline(name, item, budget, seed=0):
    """Unsupervised proxy tuning: split TRAIN into fit/val, inject synthetic anomalies
    into val, pick the config maximizing VUS-PR on that synthetic task. Uses NO test
    labels — this is the honest 'a practitioner tunes without ground truth' tier.

    Synthetic anomaly lengths are drawn log-uniformly over ~[10, len(val)/20] so the
    procedure is not biased toward any particular window scale (a fixed length would
    silently select the window size that matches it)."""
    rng = np.random.default_rng(seed)
    tr = item["train"]
    n = len(tr)
    cut = int(n * 0.7)
    fit, val = tr[:cut], tr[cut:].copy()
    if len(val) < 800:
        return None, 0.0
    lab = np.zeros(len(val), dtype=int)
    n_anom = max(4, len(val) // 1500)
    lo_len, hi_len = 10, max(20, len(val) // 20)
    scale = np.std(val, axis=0) if val.ndim > 1 else np.std(val)
    for _ in range(n_anom):
        seg_len = int(np.exp(rng.uniform(np.log(lo_len), np.log(hi_len))))
        s = int(rng.integers(0, max(1, len(val) - seg_len)))
        kind = rng.integers(0, 3)
        if kind == 0:      # spike / amplitude burst
            val[s:s + seg_len] += 4 * scale * rng.choice([-1, 1])
        elif kind == 1:    # level shift
            val[s:s + seg_len] += 2.5 * scale
        else:              # variance burst
            val[s:s + seg_len] += rng.normal(0, 3, val[s:s + seg_len].shape) * scale
        lab[s:s + seg_len] = 1
    grid = BASELINE_GRIDS[name]
    idx = rng.permutation(len(grid))[:budget]
    fn = BASELINE_FNS[name]
    best, best_cfg, t0 = -1, None, time.time()
    for i in idx:
        cfg = grid[i]
        try:
            v = all_metrics(lab, fn(fit, val, **cfg))["vus_pr"]
            if v == v and v > best:
                best, best_cfg = v, cfg
        except Exception:
            continue
    return best_cfg, time.time() - t0


def process(dataset, item, dets, tiers, path, skip, tune_budget):
    lab = item["test_labels"]
    if lab.sum() == 0:
        print(f"[skip] {item['name']}: no anomalies", flush=True)
        return

    # --- TSFM detectors: one forward pass per backend yields all usage modes ---
    tsfm_wanted = {}
    for det in dets:
        for tn, backend in TSFM_BACKENDS.items():
            for mode in ("resid", "quantile"):
                if det == f"{tn}_{mode}":
                    tsfm_wanted.setdefault(tn, []).append(mode)
    for tn, modes in tsfm_wanted.items():
        pending = [m for m in modes if (dataset, item["name"], f"{tn}_{m}", "default") not in skip]
        if not pending:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = tsfm_scores_all_modes(item["train"], item["test"], TSFM_BACKENDS[tn])
            rt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            for m in pending:
                row = base_row(dataset, item)
                row["detector"] = f"{tn}_{m}"; row["tier"] = "default"
                # shared cost across modes: report full pass time (single-mode deployment cost)
                fill(row, all_metrics(lab, out[m]), rt, peak)
                save_scores(dataset, item["name"], f"{tn}_{m}", "default", lab, out[m])
                append(path, row)
                print(f"[{dataset}] {item['name'][:34]:34s} {tn + '_' + m:24s} default  "
                      f"vus={row['vus_pr']} affil={row['affil_f1']} ({rt:.1f}s)", flush=True)
        except Exception as e:
            for m in pending:
                row = base_row(dataset, item)
                row["detector"] = f"{tn}_{m}"; row["tier"] = "default"
                row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
                append(path, row)
            traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- classical baselines across tiers ---
    for det in dets:
        if det not in BASELINE_FNS:
            continue
        for tier in tiers:
            key = (dataset, item["name"], det, tier)
            if key in skip:
                continue
            row = base_row(dataset, item); row["detector"] = det; row["tier"] = tier
            try:
                if tier == "default":
                    sc, rt, pk = run_detector(det, item["train"], item["test"])
                    fill(row, all_metrics(lab, sc), rt, pk)
                    save_scores(dataset, item["name"], det, tier, lab, sc)
                elif tier == "tuned":
                    cfg, ts = tune_baseline(det, item, tune_budget)
                    if cfg is None:
                        continue
                    t0 = time.time()
                    sc = BASELINE_FNS[det](item["train"], item["test"], **cfg)
                    fill(row, all_metrics(lab, sc), time.time() - t0, 0.0, round(ts, 2))
                    row["config"] = str(cfg)
                    save_scores(dataset, item["name"], det, tier, lab, sc)
                elif tier == "oracle":
                    grid = BASELINE_GRIDS[det]
                    best, bcfg, bm, t0 = -1, None, None, time.time()
                    for cfg in grid:
                        try:
                            m = all_metrics(lab, BASELINE_FNS[det](item["train"], item["test"], **cfg))
                            if m["vus_pr"] == m["vus_pr"] and m["vus_pr"] > best:
                                best, bcfg, bm = m["vus_pr"], cfg, m
                        except Exception:
                            continue
                    if bm is None:
                        continue
                    fill(row, bm, time.time() - t0, 0.0)
                    row["config"] = str(bcfg)
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
                traceback.print_exc()
            append(path, row)
            print(f"[{dataset}] {item['name'][:34]:34s} {det:24s} {tier:8s} "
                  f"vus={row['vus_pr']} affil={row['affil_f1']} ({row['runtime_s']}s)", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def iter_items(datasets, n_ucr, psm_len):
    """Yields (dataset, loader) in the order the datasets were requested.
    The loader is a thunk so that sharded workers do not pay the I/O cost of
    reading series they are not going to process."""
    for ds in datasets:
        if ds == "ucr":
            files = D.ucr_series_list()
            if n_ucr and n_ucr < len(files):
                idx = np.linspace(0, len(files) - 1, n_ucr).round().astype(int)
                files = [files[i] for i in idx]
            for p in files:
                yield "ucr", (lambda p=p: D.load_ucr(p))
        elif ds == "smd":
            for name in D.smd_machines():
                yield "smd", (lambda n=name: D.load_smd(n))
        elif ds == "nasa":
            for cid, sc, kind in D.nasa_channels():
                yield "nasa", (lambda c=cid: D.load_nasa(c))
        elif ds == "psm":
            yield "psm", (lambda: D.load_psm(max_len=psm_len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ucr,smd,nasa,psm")
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--tiers", default="default,tuned")
    ap.add_argument("--n-ucr", type=int, default=0, help="0 = all 250")
    ap.add_argument("--psm-len", type=int, default=40000)
    ap.add_argument("--tune-budget", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="", help="i/n — process only items where index %% n == i")
    args = ap.parse_args()

    path = csv_path(args.tag)
    skip = done_keys(path)
    dets = [d for d in args.detectors.split(",") if d]
    tiers = args.tiers.split(",")
    datasets = args.datasets.split(",")
    shard_i, shard_n = (0, 1)
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    print(f"detectors={dets}\ntiers={tiers}\ndatasets={datasets}\n"
          f"shard={shard_i}/{shard_n}\nout={path}\nalready done: {len(skip)} keys", flush=True)
    for k, (ds, loader) in enumerate(iter_items(datasets, args.n_ucr, args.psm_len)):
        if k % shard_n != shard_i:
            continue
        process(ds, loader(), dets, tiers, path, skip, args.tune_budget)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
