"""Detectors: classical baselines (with tuning) + zero-shot TSFM scorers.

TSFM usage modes (addressing the "naive usage" reviewer objection):
  - resid    : |actual - median forecast|, robust-normalized
  - quantile : degree of violation of the [q10, q90] predictive interval (calibrated)
  - embed    : Mahalanobis-style distance of TSFM encoder embeddings vs train distribution
Baselines: IsolationForest, LOF, small AE, plus MatrixProfile-style nearest-neighbour distance.
"""
import os
import time
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Setting CUDA_VISIBLE_DEVICES="" does not reliably hide the GPU on Windows: allocation
# on cuda:0 still succeeds while torch.cuda.device_count() drops to 0. That combination
# silently ran the CPU cost profile on the GPU, so device choice is an explicit override.
DEVICE = os.environ.get("KILLTEST_DEVICE") or (
    "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
MAX_CHANNELS = 12  # cost cap for multivariate TSFM scoring

# When several sharded workers run concurrently, a default of n_jobs=-1 in each worker
# oversubscribes the CPU (workers x cores threads) and slows everything down.
# KILLTEST_N_JOBS caps the per-worker parallelism.
N_JOBS = int(os.environ.get("KILLTEST_N_JOBS", "-1"))
if os.environ.get("KILLTEST_TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["KILLTEST_TORCH_THREADS"]))


# ---------------- windowing ----------------

def sliding_windows(x, w, stride=1):
    if x.ndim == 1:
        x = x[:, None]
    T, C = x.shape
    n = max(1, (T - w) // stride + 1)
    idx = np.arange(w)[None, :] + stride * np.arange(n)[:, None]
    idx = np.clip(idx, 0, T - 1)
    return x[idx].reshape(n, w * C), idx


def windows_to_points(win_scores, idx, T):
    s = np.zeros(T); cnt = np.zeros(T)
    np.add.at(s, idx.ravel(), np.repeat(win_scores, idx.shape[1]))
    np.add.at(cnt, idx.ravel(), 1)
    cnt[cnt == 0] = 1
    return s / cnt


def robust_norm(x, ref):
    med = np.median(ref, axis=0)
    iqr = np.subtract(*np.percentile(ref, [75, 25], axis=0)) + 1e-8
    return (x - med) / iqr


def _prep(train, test):
    return robust_norm(train, train), robust_norm(test, train)


# ---------------- classical baselines ----------------

def score_iforest(train, test, w=100, n_estimators=200, max_samples="auto", seed=0):
    tr, te = _prep(train, test)
    tr_w, _ = sliding_windows(tr, w, stride=max(1, w // 20))
    te_w, te_idx = sliding_windows(te, w, stride=1)
    clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                          random_state=seed, n_jobs=N_JOBS).fit(tr_w)
    return windows_to_points(-clf.score_samples(te_w), te_idx, len(test))


def score_lof(train, test, w=100, k=50):
    tr, te = _prep(train, test)
    tr_w, _ = sliding_windows(tr, w, stride=max(1, w // 20))
    te_w, te_idx = sliding_windows(te, w, stride=max(1, w // 50))
    clf = LocalOutlierFactor(n_neighbors=min(k, max(2, len(tr_w) - 1)), novelty=True, n_jobs=N_JOBS).fit(tr_w)
    return windows_to_points(-clf.score_samples(te_w), te_idx, len(test))


def score_knn(train, test, w=100, k=5):
    """Nearest-neighbour distance to training windows (matrix-profile-like)."""
    tr, te = _prep(train, test)
    tr_w, _ = sliding_windows(tr, w, stride=max(1, w // 20))
    te_w, te_idx = sliding_windows(te, w, stride=max(1, w // 50))
    nn = NearestNeighbors(n_neighbors=min(k, len(tr_w)), n_jobs=N_JOBS).fit(tr_w)
    d, _ = nn.kneighbors(te_w)
    return windows_to_points(d.mean(1), te_idx, len(test))


class _AE(torch.nn.Module):
    def __init__(self, d, h=64, z=16):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, z))
        self.dec = torch.nn.Sequential(torch.nn.Linear(z, h), torch.nn.ReLU(), torch.nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def score_ae(train, test, w=100, epochs=30, hidden=64, latent=16, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    tr, te = _prep(train, test)
    tr_w, _ = sliding_windows(tr, w, stride=max(1, w // 20))
    te_w, te_idx = sliding_windows(te, w, stride=1)
    sc = StandardScaler().fit(tr_w)
    trt = torch.tensor(sc.transform(tr_w), dtype=torch.float32, device=DEVICE)
    tet = torch.tensor(sc.transform(te_w), dtype=torch.float32, device=DEVICE)
    model = _AE(trt.shape[1], hidden, latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(trt), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad(); torch.nn.functional.mse_loss(model(xb), xb).backward(); opt.step()
    errs = []
    with torch.no_grad():
        for i in range(0, len(tet), 4096):
            b = tet[i:i + 4096]
            errs.append(((model(b) - b) ** 2).mean(1).cpu().numpy())
    return windows_to_points(np.concatenate(errs), te_idx, len(test))


# ---------------- TSFM backends ----------------

_CACHE = {}


def _chronos(name):
    if name not in _CACHE:
        from chronos import BaseChronosPipeline
        _CACHE[name] = BaseChronosPipeline.from_pretrained(name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    return _CACHE[name]


def _timesfm():
    if "timesfm" not in _CACHE:
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        # compile() derives global_batch_size from model.device_count, which is 0 whenever
        # CUDA is masked -- that makes forecast() assert instead of running on CPU. Pin the
        # placement explicitly; device_count was already 1 on GPU, so GPU runs are unchanged.
        m.model.to(DEVICE)
        m.model.device = torch.device(DEVICE)
        m.model.device_count = 1
        # per_core_batch_size defaults to 1 and makes forecasting ~70x slower; batch it.
        m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=256,
                                         normalize_inputs=True, use_continuous_quantile_head=True,
                                         per_core_batch_size=64))
        _CACHE["timesfm"] = m
    return _CACHE["timesfm"]


def _tirex():
    if "tirex" not in _CACHE:
        from tirex import load_model
        # device defaults to CUDA; the checkpoint then deserializes onto cuda:0 and fails
        # when the GPU is masked.
        _CACHE["tirex"] = load_model("NX-AI/TiRex", device=DEVICE)
    return _CACHE["tirex"]


def _forecast_quantiles(backend, contexts, horizon, levels=(0.1, 0.5, 0.9)):
    """Returns array (B, horizon, len(levels)). Contexts: list of 1-D float arrays."""
    if backend.startswith("chronos"):
        pipe = _chronos(backend.replace("chronos:", ""))
        q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in contexts],
                                      prediction_length=horizon, quantile_levels=list(levels))
        return q.float().cpu().numpy()
    if backend == "timesfm":
        m = _timesfm()
        _, q = m.forecast(horizon=horizon, inputs=[np.asarray(c, dtype=np.float32) for c in contexts])
        q = np.asarray(q)  # (B, H, 10): [mean, q0.1..q0.9]
        cols = [int(round(l * 10)) for l in levels]  # q0.1 -> col 1, q0.5 -> 5, q0.9 -> 9
        return q[:, :, cols]
    if backend == "tirex":
        m = _tirex()
        maxlen = max(len(c) for c in contexts)
        padded = np.stack([np.pad(np.asarray(c, dtype=np.float32), (maxlen - len(c), 0), mode="edge")
                           for c in contexts])
        q, _ = m.forecast(context=torch.tensor(padded), prediction_length=horizon)
        q = q.float().cpu().numpy()  # (B, H, 9) for deciles q0.1..q0.9
        cols = [int(round(l * 10)) - 1 for l in levels]
        return q[:, :, cols]
    raise ValueError(backend)


def _rolling_contexts(full, test_start, context, horizon):
    starts = list(range(test_start, len(full), horizon))
    ctxs = []
    for s in starts:
        c = full[max(0, s - context):s]
        if len(c) < 32:
            c = np.pad(c, (32 - len(c), 0), mode="edge")
        ctxs.append(c)
    return starts, ctxs


def _score_from_quantiles(full, test_start, starts, q, horizon):
    """q: (n_starts, horizon, 3) for levels (0.1, 0.5, 0.9). Returns (resid, quantile-violation)."""
    T = len(full)
    n_test = T - test_start
    resid = np.zeros(n_test); viol = np.zeros(n_test)
    for j, s in enumerate(starts):
        e = min(s + horizon, T)
        h = e - s
        act = full[s:e]
        lo, med, hi = q[j, :h, 0], q[j, :h, 1], q[j, :h, 2]
        resid[s - test_start:e - test_start] = np.abs(act - med)
        width = np.maximum(hi - lo, 1e-8)
        viol[s - test_start:e - test_start] = np.maximum(np.maximum(lo - act, act - hi), 0) / width
    ref = np.abs(np.diff(full[:test_start]))
    scale = np.percentile(ref, 90) + 1e-8 if len(ref) else 1.0
    return resid / scale, viol


def tsfm_scores_all_modes(train, test, backend, context=512, horizon=64, batch=256):
    """Compute resid and quantile scores in a single pass, batching across channels.

    Returns {"resid": scores, "quantile": scores}. Multivariate series are scored
    per top-variance channel and aggregated by the mean of the top quartile of channels.
    """
    uni = train.ndim == 1
    if uni:
        chans = [None]
        fulls = [np.concatenate([train, test])]
    else:
        var = train.var(axis=0)
        chans = np.argsort(var)[::-1][:min(train.shape[1], MAX_CHANNELS)]
        fulls = [np.concatenate([train[:, c], test[:, c]]) for c in chans]
    test_start = len(train)

    # collect contexts across all channels, forecast in large batches
    all_ctx, owner = [], []
    per_chan_starts = []
    for ci, full in enumerate(fulls):
        starts, ctxs = _rolling_contexts(full, test_start, context, horizon)
        per_chan_starts.append(starts)
        all_ctx.extend(ctxs)
        owner.extend([ci] * len(ctxs))
    qs = []
    for i in range(0, len(all_ctx), batch):
        qs.append(_forecast_quantiles(backend, all_ctx[i:i + batch], horizon, levels=(0.1, 0.5, 0.9)))
    Q = np.concatenate(qs, axis=0)
    owner = np.asarray(owner)

    resid_cols, viol_cols = [], []
    for ci, full in enumerate(fulls):
        qi = Q[owner == ci]
        r, v = _score_from_quantiles(full, test_start, per_chan_starts[ci], qi, horizon)
        resid_cols.append(r); viol_cols.append(v)

    def agg(cols):
        if uni:
            return cols[0]
        M = np.stack(cols, axis=1)
        k = max(1, M.shape[1] // 4)
        return np.sort(M, axis=1)[:, -k:].mean(axis=1)

    return {"resid": agg(resid_cols), "quantile": agg(viol_cols)}


def score_tsfm(train, test, backend="chronos:amazon/chronos-bolt-small", mode="resid",
               context=512, horizon=64):
    return tsfm_scores_all_modes(train, test, backend, context, horizon)[mode]


# ---------------- registry ----------------

TSFM_BACKENDS = {
    "chronos_small": "chronos:amazon/chronos-bolt-small",
    "chronos_base": "chronos:amazon/chronos-bolt-base",
    "timesfm": "timesfm",
    "tirex": "tirex",
}

BASELINE_FNS = {
    "iforest": score_iforest,
    "lof": score_lof,
    "knn": score_knn,
    "ae": score_ae,
}

# tuning grids (budgeted random search; cost recorded for break-even analysis)
BASELINE_GRIDS = {
    "iforest": [{"w": w, "n_estimators": n} for w in (50, 100, 200, 400) for n in (100, 200)],
    "lof": [{"w": w, "k": k} for w in (50, 100, 200, 400) for k in (10, 30, 50)],
    "knn": [{"w": w, "k": k} for w in (50, 100, 200, 400) for k in (1, 3, 5)],
    "ae": [{"w": w, "hidden": h, "latent": z} for w in (50, 100, 200) for h in (32, 64) for z in (8, 16)],
}


def build_detectors(tsfm_modes=("resid", "quantile")):
    d = {}
    for bn, fn in BASELINE_FNS.items():
        d[bn] = (lambda fn: lambda tr, te: fn(tr, te))(fn)
    for tn, backend in TSFM_BACKENDS.items():
        for mode in tsfm_modes:
            d[f"{tn}_{mode}"] = (lambda b, m: lambda tr, te: score_tsfm(tr, te, b, m))(backend, mode)
    return d


DETECTORS = build_detectors()


def run_detector(name, train, test):
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    scores = DETECTORS[name](train, test)
    peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    return scores, time.time() - t0, peak
