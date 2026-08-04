"""Data loaders for UCR Anomaly Archive, SMD, SMAP/MSL, PSM."""
import ast
import csv
import re
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
UCR_DIR = DATA / "UCR" / "AnomalyDatasets_2021" / "UCR_TimeSeriesAnomalyDatasets2021" / "FilesAreInHere" / "UCR_Anomaly_FullData"
SMD_DIR = DATA / "SMD" / "OmniAnomaly-master" / "ServerMachineDataset"
NASA_DIR = DATA / "NASA" / "TranAD-main" / "data" / "SMAP_MSL"
PSM_DIR = DATA / "PSM"

UCR_RE = re.compile(r"_(\d+)_(\d+)_(\d+)\.txt$")


def ucr_series_list():
    return sorted(UCR_DIR.glob("*.txt"))


def ucr_killtest_subset(n=40):
    """Deterministic stride subset over the sorted 250 files — no cherry-picking."""
    files = ucr_series_list()
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx]


def load_ucr(path):
    m = UCR_RE.search(path.name)
    train_end, a_start, a_end = map(int, m.groups())
    x = np.loadtxt(path)
    labels = np.zeros(len(x), dtype=int)
    labels[a_start:a_end + 1] = 1
    return {
        "name": path.stem,
        "train": x[:train_end],
        "test": x[train_end:],
        "test_labels": labels[train_end:],
        "domain": path.stem.split("_")[3] if len(path.stem.split("_")) > 3 else "unknown",
    }


def smd_machines(every=1):
    names = sorted(p.stem for p in (SMD_DIR / "test").glob("*.txt"))
    return names[::every]


def load_smd(name):
    tr = np.loadtxt(SMD_DIR / "train" / f"{name}.txt", delimiter=",")
    te = np.loadtxt(SMD_DIR / "test" / f"{name}.txt", delimiter=",")
    lab = np.loadtxt(SMD_DIR / "test_label" / f"{name}.txt", delimiter=",").astype(int)
    return {"name": name, "train": tr, "test": te, "test_labels": lab, "domain": "server"}


# ---------- SMAP / MSL ----------

def _nasa_meta():
    with open(NASA_DIR / "labeled_anomalies.csv") as f:
        return list(csv.DictReader(f))


def nasa_channels(spacecraft=None):
    """Returns list of (chan_id, spacecraft, anomaly_class)."""
    out = []
    for r in _nasa_meta():
        if spacecraft and r["spacecraft"] != spacecraft:
            continue
        if not (NASA_DIR / "test" / f'{r["chan_id"]}.npy').exists():
            continue
        classes = r["class"].strip("[]").replace(" ", "").split(",")
        kind = "mixed" if len(set(classes)) > 1 else classes[0]
        out.append((r["chan_id"], r["spacecraft"], kind))
    return out


def load_nasa(chan_id):
    r = next(x for x in _nasa_meta() if x["chan_id"] == chan_id)
    tr = np.load(NASA_DIR / "train" / f"{chan_id}.npy")
    te = np.load(NASA_DIR / "test" / f"{chan_id}.npy")
    lab = np.zeros(len(te), dtype=int)
    for s, e in ast.literal_eval(r["anomaly_sequences"]):
        lab[s:e + 1] = 1
    classes = r["class"].strip("[]").replace(" ", "").split(",")
    kind = "mixed" if len(set(classes)) > 1 else classes[0]
    return {"name": chan_id, "train": tr, "test": te, "test_labels": lab,
            "domain": r["spacecraft"].lower(), "anomaly_kind": kind}


# ---------- PSM ----------

def load_psm(max_len=None):
    tr = np.genfromtxt(PSM_DIR / "train.csv", delimiter=",", skip_header=1)[:, 1:]
    te = np.genfromtxt(PSM_DIR / "test.csv", delimiter=",", skip_header=1)[:, 1:]
    lab = np.genfromtxt(PSM_DIR / "test_label.csv", delimiter=",", skip_header=1)[:, 1].astype(int)
    tr = np.nan_to_num(tr); te = np.nan_to_num(te)
    if max_len:
        tr, te, lab = tr[:max_len], te[:max_len], lab[:max_len]
    return {"name": "psm", "train": tr, "test": te, "test_labels": lab, "domain": "server_app"}
