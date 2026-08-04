"""Kill-test runner: UCR 40-series subset + SMD subset, 5 detectors, incremental CSV output.
Usage: python -m killtest.run [--only ucr|smd] [--detectors a,b,c]
"""
import argparse, csv, gc, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

from . import data as D
from .detectors import run_detector, DETECTORS
from .metrics import all_metrics

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
CSV = OUT / "killtest_results.csv"
FIELDS = ["dataset", "series", "domain", "detector", "auc_pr", "vus_pr", "event_f1",
          "runtime_s", "n_test", "anomaly_frac", "status"]


def done_keys():
    if not CSV.exists():
        return set()
    with open(CSV) as f:
        return {(r["dataset"], r["series"], r["detector"]) for r in csv.DictReader(f)}


def append(row):
    new = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def process(dataset, item, detectors, skip):
    labels = item["test_labels"]
    for det in detectors:
        key = (dataset, item["name"], det)
        if key in skip:
            continue
        row = dict(dataset=dataset, series=item["name"], domain=item["domain"], detector=det,
                   n_test=len(labels), anomaly_frac=round(float(np.mean(labels)), 5),
                   auc_pr="", vus_pr="", event_f1="", runtime_s="", status="ok")
        try:
            scores, rt = run_detector(det, item["train"], item["test"])
            m = all_metrics(labels, scores)
            row.update({k: round(v, 5) if v == v else "" for k, v in m.items()},
                       runtime_s=round(rt, 2))
        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {e}"
            traceback.print_exc()
        append(row)
        print(f"[{dataset}] {item['name']} :: {det} -> {row.get('vus_pr','ERR')} ({row['runtime_s']}s)", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ucr", "smd"], default=None)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--n-ucr", type=int, default=40)
    args = ap.parse_args()
    dets = args.detectors.split(",")
    skip = done_keys()

    if args.only in (None, "ucr"):
        for p in D.ucr_killtest_subset(args.n_ucr):
            process("ucr", D.load_ucr(p), dets, skip)
    if args.only in (None, "smd"):
        for name in D.smd_machines(every=5):
            process("smd", D.load_smd(name), dets, skip)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
