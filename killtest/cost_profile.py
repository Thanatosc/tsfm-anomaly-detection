"""Isolated cost profiling: wall-clock, throughput, and peak memory per method.

Each (device, detector, series) measurement runs in a FRESH subprocess so that
peak GPU memory is not contaminated by other models resident in the same process
(this is why the in-run `peak_mb` column of full_results.csv is not usable for
memory comparisons; use this file instead).

Run this on an OTHERWISE IDLE machine. The v1 profile was collected while analysis
jobs ran concurrently and every throughput figure came out 2.9-5.6x low, unevenly
across detectors -- enough to distort their relative ranking. `other_py` records the
concurrent python-process count per row so that condition is visible rather than silent.
"""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cost_profile.csv"
FIELDS = ["device", "detector", "dataset", "series", "n_test", "n_channels",
          "runtime_s", "cold_s", "points_per_s", "peak_mb", "rss_mb", "other_py", "status"]


def _other_python_procs():
    """Count python processes other than this one -- a contention canary.

    Baseline during a healthy run is 3, not 0: the Windows venv python.exe is a launcher
    stub that re-execs the real interpreter, so the orchestrator and this child each show
    up twice (orchestrator stub + orchestrator real + this child's own stub). Readings
    above ~3 mean something else is competing for the CPU and the row is suspect.
    """
    try:
        import psutil
        me = os.getpid()
        return sum(1 for p in psutil.process_iter(["name", "pid"])
                   if (p.info["name"] or "").lower().startswith("python") and p.info["pid"] != me)
    except Exception:
        return -1


# ---------------- child process entry ----------------

def _child(dataset, series, detector, device):
    # NOT CUDA_VISIBLE_DEVICES="": on Windows that leaves cuda:0 allocatable while zeroing
    # device_count(), which silently profiled "cpu" runs on the GPU. detectors.DEVICE reads
    # this override at import time, so it must be set before importing them.
    os.environ["KILLTEST_DEVICE"] = device
    import numpy as np, torch
    from . import data as D
    from .detectors import BASELINE_FNS, TSFM_BACKENDS, tsfm_scores_all_modes

    item = {"ucr": lambda s: D.load_ucr(next(p for p in D.ucr_series_list() if p.stem == s)),
            "smd": D.load_smd,
            "nasa": D.load_nasa}[dataset](series)
    n = len(item["test"])
    nc = 1 if item["test"].ndim == 1 else item["test"].shape[1]

    def score():
        if detector in BASELINE_FNS:
            BASELINE_FNS[detector](item["train"], item["test"])
        else:
            tsfm_scores_all_modes(item["train"], item["test"], TSFM_BACKENDS[detector])

    on_cuda = device == "cuda" and torch.cuda.is_available()
    # cold: includes model download-from-cache + weight load (batch-job cost)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); score(); cold = time.time() - t0
    # warm: model already resident (streaming / repeated-inference cost)
    others = _other_python_procs()
    t0 = time.time(); score(); warm = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1e6 if on_cuda else 0.0
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e6
    except Exception:
        rss = 0.0
    print("RESULT" + json.dumps({"n_test": n, "n_channels": nc,
                                 "runtime_s": round(warm, 3), "cold_s": round(cold, 3),
                                 "points_per_s": round(n / max(warm, 1e-6), 1),
                                 "peak_mb": round(peak, 1), "rss_mb": round(rss, 1),
                                 "other_py": others}))


# ---------------- parent ----------------

def measure(dataset, series, detector, device, timeout=None):
    # A single CPU measurement can legitimately run for over an hour: TiRex costs ~411 ms
    # per 512-point context and cannot parallelise across cores (sLSTM recurrence), so
    # SMD machine-1-1 at 12 channels is ~73 min for the cold+warm pair. The original flat
    # 3600 s cap silently killed exactly those rows after burning the full hour.
    if timeout is None:
        timeout = 6 * 3600 if device == "cpu" else 3600
    env = dict(os.environ, HF_HUB_DISABLE_XET="1", PYTHONPATH=str(ROOT))
    cmd = [sys.executable, "-m", "killtest.cost_profile", "--child",
           "--dataset", dataset, "--series", series, "--detector", detector, "--device", device]
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT"):
                return json.loads(line[6:]), "ok"
        return {}, f"error: no result (rc={p.returncode}) {p.stderr.strip()[-160:]}"
    except subprocess.TimeoutExpired:
        return {}, "error: timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--dataset"); ap.add_argument("--series")
    ap.add_argument("--detector"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--n-per-dataset", type=int, default=2)
    args = ap.parse_args()

    if args.child:
        _child(args.dataset, args.series, args.detector, args.device)
        return

    from . import data as D
    from .detectors import BASELINE_FNS, TSFM_BACKENDS

    targets = []
    for p in D.ucr_killtest_subset(args.n_per_dataset):
        targets.append(("ucr", p.stem))
    for m in D.smd_machines()[: args.n_per_dataset]:
        targets.append(("smd", m))
    for cid, _, _ in D.nasa_channels()[: args.n_per_dataset]:
        targets.append(("nasa", cid))
    detectors = list(BASELINE_FNS) + list(TSFM_BACKENDS)

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {(r["device"], r["detector"], r["dataset"], r["series"]) for r in csv.DictReader(f)}
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for device in args.devices.split(","):
            for ds, series in targets:
                for det in detectors:
                    if (device, det, ds, series) in done:
                        continue
                    res, status = measure(ds, series, det, device)
                    row = dict(device=device, detector=det, dataset=ds, series=series,
                               n_test=res.get("n_test", ""), n_channels=res.get("n_channels", ""),
                               runtime_s=res.get("runtime_s", ""), cold_s=res.get("cold_s", ""),
                               points_per_s=res.get("points_per_s", ""),
                               peak_mb=res.get("peak_mb", ""), rss_mb=res.get("rss_mb", ""),
                               other_py=res.get("other_py", ""), status=status)
                    w.writerow(row); f.flush()
                    print(f"{device:5s} {det:16s} {ds:5s} {series[:26]:26s} "
                          f"warm={row['runtime_s']}s cold={row['cold_s']}s "
                          f"{row['points_per_s']}pts/s {row['peak_mb']}MB "
                          f"others={row['other_py']} {status}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
