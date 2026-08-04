"""Check experiment progress. Run:  .venv\\Scripts\\python.exe check_progress.py"""
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESDIR = ROOT / "results"
LOG = RESDIR / "run_full.log"

EXPECTED = {"ucr": 250, "smd": 28, "nasa": 81, "psm": 1}  # nasa: 82 metadata rows, P-2 listed twice


def running():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*run_full*' }).Count"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0)
    except Exception:
        return -1


def main():
    files = sorted(RESDIR.glob("full_results*.csv"))
    if not files:
        print("no results yet"); return
    done = Counter()
    rows = errors = 0
    seen_rows, seen_series = set(), set()
    for p in files:
        with open(p) as f:
            for r in csv.DictReader(f):
                k = (r["dataset"], r["series"], r["detector"], r["tier"])
                if k in seen_rows:
                    continue
                seen_rows.add(k)
                rows += 1
                if r["status"] != "ok":
                    errors += 1
                s = (r["dataset"], r["series"])
                if s not in seen_series:
                    seen_series.add(s)
                    done[r["dataset"]] += 1

    print(f"shard files  : {', '.join(p.name for p in files)}")
    print(f"rows written : {rows}   (errors: {errors})")
    print("series completed per dataset:")
    total_done = total_exp = 0
    for ds, exp in EXPECTED.items():
        d = done.get(ds, 0)
        total_done += d; total_exp += exp
        filled = int(24 * min(d, exp) / exp)
        print(f"  {ds:5s} {d:4d}/{exp:<4d} [{'#' * filled}{'.' * (24 - filled)}]")
    pct = 100 * total_done / total_exp
    print(f"overall      : {total_done}/{total_exp} series ({pct:.1f}%)")

    n = running()
    print(f"workers      : {n if n >= 0 else '?'} running")
    if n == 0 and total_done >= total_exp:
        print("\n>>> EXPERIMENT FINISHED")
    elif n == 0:
        print("\n>>> STOPPED BEFORE FINISHING — run run_experiment.bat to resume "
              "(completed work is skipped automatically)")
    else:
        print("\n>>> still running")


if __name__ == "__main__":
    main()
