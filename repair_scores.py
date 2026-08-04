"""Repair the persisted score archive (peer review F2).

`save_scores` stored float16, which saturates at 65504. 1213 of 5504 arrays (22%) hold inf
as a result, concentrated in NASA and SMD where k-NN/AE distances and TSFM residuals are
large. `full_results.csv` is unaffected because its metrics were computed on float64 before
saving, but every analysis recomputed *from the archive* (the §6.3 mechanism analysis, the
score-aggregation control) inherits the corruption, and the matched fully-finite subset is
almost pure UCR, which would silently turn §6.3 into a univariate-only result.

This recomputes only the affected (dataset, series, detector, tier) combinations and
re-persists them as float32. Baseline configurations are read back from the `config` column
rather than re-tuned, so the reproduction is exact rather than merely equivalent. Each
recomputed array's VUS-PR is checked against the archived CSV value; a mismatch means the
score is not reproducing, which is a finding in its own right and is reported.

Run: .venv/Scripts/python.exe repair_scores.py [--dry-run] [--only classical|tsfm]
"""
import argparse
import ast
import csv
import glob
import sys
import time
from collections import defaultdict

import numpy as np

from killtest import data as D
from killtest.detectors import BASELINE_FNS, TSFM_BACKENDS, run_detector, tsfm_scores_all_modes
from killtest.metrics import all_metrics
from killtest.run_full import SCORES, save_scores

TSFM_PREFIX = ("chronos", "timesfm", "tirex")


def is_tsfm(det):
    return det.startswith(TSFM_PREFIX)


def npz_bad(dataset, series, detector, tier):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in series)
    p = SCORES / f"{dataset}__{safe}__{detector}__{tier}.npz"
    if not p.exists():
        return True
    try:
        with np.load(p) as z:
            return not np.isfinite(z["scores"].astype(np.float64)).all()
    except Exception:
        return True


def load_rows():
    rows, seen = [], set()
    for f in sorted(glob.glob("results/full_results_*.csv")):
        for r in csv.DictReader(open(f)):
            k = (r["dataset"], r["series"], r["detector"], r["tier"])
            if k in seen or r.get("status") not in (None, "", "ok"):
                continue
            seen.add(k)
            rows.append(r)
    return rows


def load_item(dataset, series, ucr_index):
    if dataset == "ucr":
        return D.load_ucr(ucr_index[series])
    if dataset == "smd":
        return D.load_smd(series)
    if dataset == "nasa":
        return D.load_nasa(series)
    return D.load_psm(max_len=40000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["classical", "tsfm"], default=None)
    args = ap.parse_args()

    rows = load_rows()
    todo = [r for r in rows if npz_bad(r["dataset"], r["series"], r["detector"], r["tier"])]
    if args.only == "classical":
        todo = [r for r in todo if not is_tsfm(r["detector"])]
    elif args.only == "tsfm":
        todo = [r for r in todo if is_tsfm(r["detector"])]

    by_series = defaultdict(list)
    for r in todo:
        by_series[(r["dataset"], r["series"])].append(r)

    n_tsfm = sum(1 for r in todo if is_tsfm(r["detector"]))
    print(f"{len(todo)} arrays to repair across {len(by_series)} series "
          f"({n_tsfm} TSFM, {len(todo) - n_tsfm} classical)", flush=True)
    if args.dry_run:
        return

    ucr_index = {p.stem: p for p in D.ucr_series_list()}
    ok = mismatch = failed = 0
    t_start = time.time()

    for i, (key, group) in enumerate(sorted(by_series.items()), 1):
        dataset, series = key
        try:
            item = load_item(dataset, series, ucr_index)
        except Exception as e:
            print(f"[{i}/{len(by_series)}] {dataset}/{series}: LOAD FAILED {e}", flush=True)
            failed += len(group)
            continue
        lab = item["test_labels"]

        # TSFM: one forward pass per backend yields both usage modes
        backends = defaultdict(list)
        for r in group:
            if is_tsfm(r["detector"]):
                for tn in TSFM_BACKENDS:
                    for mode in ("resid", "quantile"):
                        if r["detector"] == f"{tn}_{mode}":
                            backends[tn].append((mode, r))
        for tn, entries in backends.items():
            try:
                out = tsfm_scores_all_modes(item["train"], item["test"], TSFM_BACKENDS[tn])
            except Exception as e:
                print(f"[{i}] {dataset}/{series} {tn}: FAILED {e}", flush=True)
                failed += len(entries)
                continue
            for mode, r in entries:
                sc = out[mode]
                got, want = all_metrics(lab, sc)["vus_pr"], float(r["vus_pr"])
                save_scores(dataset, series, f"{tn}_{mode}", "default", lab, sc)
                if abs(got - want) > 5e-3:
                    print(f"    MISMATCH {dataset}/{series} {tn}_{mode}: "
                          f"csv={want:.4f} recomputed={got:.4f}", flush=True)
                    mismatch += 1
                else:
                    ok += 1

        # classical: reproduce the archived configuration exactly
        for r in group:
            det, tier = r["detector"], r["tier"]
            if is_tsfm(det) or det not in BASELINE_FNS:
                continue
            try:
                if tier == "default":
                    sc, _, _ = run_detector(det, item["train"], item["test"])
                else:
                    cfg = ast.literal_eval(r["config"]) if r.get("config") else {}
                    sc = BASELINE_FNS[det](item["train"], item["test"], **cfg)
            except Exception as e:
                print(f"[{i}] {dataset}/{series} {det}/{tier}: FAILED {e}", flush=True)
                failed += 1
                continue
            got, want = all_metrics(lab, sc)["vus_pr"], float(r["vus_pr"])
            save_scores(dataset, series, det, tier, lab, sc)
            if abs(got - want) > 5e-3:
                print(f"    MISMATCH {dataset}/{series} {det}/{tier}: "
                      f"csv={want:.4f} recomputed={got:.4f}", flush=True)
                mismatch += 1
            else:
                ok += 1

        if i % 10 == 0 or i == len(by_series):
            el = time.time() - t_start
            print(f"[{i}/{len(by_series)}] {el/60:.1f} min elapsed, "
                  f"ok={ok} mismatch={mismatch} failed={failed}", flush=True)

    print(f"\nDONE  reproduced={ok}  mismatched={mismatch}  failed={failed}", flush=True)
    if mismatch:
        print("MISMATCHES ARE A FINDING: the archive does not reproduce for those rows.",
              flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
