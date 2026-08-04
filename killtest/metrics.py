"""Evaluation metrics for time-series anomaly detection.

Implements:
  - AUC-PR (point-wise)
  - R-AUC-PR / VUS-PR following Paparrizos et al. (VLDB 2022) with a ramp-shaped
    label transform over a buffer of length ell, and VUS-PR = mean over ell grid.
  - Affiliation-based precision/recall F1 (Huet et al., KDD 2022), distance-based.
  - Event-level F1 with PA%K (Kim et al., AAAI 2022) averaged over K grid.

Note: we deliberately do NOT report classic point-adjusted F1 as a headline metric
(see arXiv:2607.11969 for why post-PA metrics remain fragile).
"""
import numpy as np
from sklearn.metrics import average_precision_score


# ---------------- segment utilities ----------------

def segments(labels):
    """Contiguous 1-runs as list of (start, end_inclusive)."""
    lab = np.asarray(labels).astype(int)
    d = np.diff(np.concatenate([[0], lab, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------- point-wise AP ----------------

def auc_pr(labels, scores):
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return float(average_precision_score(labels, scores))


# ---------------- range / VUS ----------------

def _ramp_labels(labels, ell):
    """Continuous label in [0,1]: 1 inside anomaly, linearly decaying to 0 over
    a buffer of ell points on each side (Paparrizos et al. 2022)."""
    lab = np.asarray(labels).astype(float)
    if ell <= 0:
        return lab
    out = lab.copy()
    n = len(lab)
    for s, e in segments(labels):
        for j in range(1, ell + 1):
            w = 1.0 - j / (ell + 1)
            if s - j >= 0:
                out[s - j] = max(out[s - j], w)
            if e + j < n:
                out[e + j] = max(out[e + j], w)
    return out


def range_pr_auc(labels, scores, ell, n_thresholds=200):
    """Range-based AUC-PR with ramp labels: existence-aware recall + overlap-weighted precision.

    Uses the continuous-label formulation: for threshold t with prediction P,
      recall_r    = sum(ramp[P]) / sum(ramp)          (existence+overlap credit)
      precision_r = sum(ramp[P]) / |P|                (partial credit for near-misses)
    then integrates precision over recall (AP-style, no interpolation).
    """
    scores = np.asarray(scores, dtype=float)
    ramp = _ramp_labels(labels, ell)
    total = ramp.sum()
    if total <= 0:
        return np.nan
    qs = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds)))
    prs = []
    for t in qs[::-1]:
        pred = scores >= t
        k = int(pred.sum())
        if k == 0:
            continue
        hit = float(ramp[pred].sum())
        prs.append((hit / total, hit / k))  # (recall, precision)
    if not prs:
        return np.nan
    prs.sort(key=lambda x: x[0])
    rec = np.array([0.0] + [p[0] for p in prs])
    prec = np.array([prs[0][1]] + [p[1] for p in prs])
    return float(np.sum(np.diff(rec) * prec[1:]))


def vus_pr(labels, scores, ell_max=None, n_ell=8):
    """Volume Under the Surface for PR: mean of range_pr_auc over an ell grid.
    ell_max defaults to the mean anomaly segment length (capped), per VUS practice."""
    segs = segments(labels)
    if not segs:
        return np.nan
    if ell_max is None:
        mean_len = int(np.mean([e - s + 1 for s, e in segs]))
        ell_max = int(np.clip(mean_len, 4, 200))
    grid = np.unique(np.linspace(0, ell_max, n_ell).round().astype(int))
    vals = [range_pr_auc(labels, scores, int(l)) for l in grid]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else np.nan


# ---------------- affiliation F1 ----------------

def _affiliation_zones(segs, n):
    """Each ground-truth event owns the points closer to it than to any other event."""
    if len(segs) == 1:
        return [(0, n - 1)]
    zones = []
    for i, (s, e) in enumerate(segs):
        lo = 0 if i == 0 else (segs[i - 1][1] + s) // 2 + 1
        hi = n - 1 if i == len(segs) - 1 else (e + segs[i + 1][0]) // 2
        zones.append((lo, hi))
    return zones


def _mean_dist_to_set(points, targets_ranges):
    """Mean distance from each point in `points` to the nearest point covered by targets."""
    if len(points) == 0:
        return 0.0
    tgt = np.concatenate([np.arange(s, e + 1) for s, e in targets_ranges]) if targets_ranges else None
    if tgt is None or len(tgt) == 0:
        return np.inf
    idx = np.searchsorted(tgt, points)
    idx = np.clip(idx, 1, len(tgt) - 1)
    left, right = tgt[idx - 1], tgt[np.clip(idx, 0, len(tgt) - 1)]
    return float(np.mean(np.minimum(np.abs(points - left), np.abs(points - right))))


MAX_PREDICTED_RATE = 0.30  # a detector flagging >30% of points is not deployable;
                           # such thresholds are excluded from every best-over-threshold sweep.


def _threshold_grid(scores, n_thresholds, lo=0.5, hi=0.9995):
    """Candidate thresholds, excluding those that would flag an implausible
    fraction of the series. Without this guard a constant score collapses the
    sweep onto 'predict everything', which saturates event- and affiliation-based
    metrics (see killtest/test_metrics.py)."""
    qs = np.unique(np.quantile(scores, np.linspace(lo, hi, n_thresholds)))
    n = len(scores)
    return [t for t in qs if (scores >= t).sum() <= MAX_PREDICTED_RATE * n]


def affiliation_f1(labels, scores, n_thresholds=60):
    """Best-over-threshold affiliation F1 using the reference implementation of
    Huet et al. (KDD 2022), package `affiliation` (github.com/ahstat/affiliation-metrics-py).

    NOTE: this metric has a high floor — a random detector scores ~0.67 on a long
    series with a single event, because affiliation credit is distance-based. Always
    read it against the random baseline, never as an absolute quality level.
    Returns NaN if the package is unavailable."""
    try:
        from affiliation.generics import convert_vector_to_events
        from affiliation.metrics import pr_from_events
    except ImportError:
        return np.nan
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    ev_gt = convert_vector_to_events(labels)
    if not ev_gt:
        return np.nan
    n = len(labels)
    best = 0.0
    for t in _threshold_grid(scores, n_thresholds):
        pred = (scores >= t).astype(int)
        ev_pred = convert_vector_to_events(pred)
        if not ev_pred:
            continue
        try:
            res = pr_from_events(ev_pred, ev_gt, (0, n))
        except Exception:
            continue
        p, r = res["precision"], res["recall"]
        if p is None or r is None or p != p or r != r:
            continue
        if p + r > 0:
            best = max(best, 2 * p * r / (p + r))
    return float(best)


# ---------------- event F1 with PA%K ----------------

def event_f1_pak(labels, scores, ks=(0.0, 0.2, 0.5, 0.8), n_thresholds=100):
    """Best-over-threshold event-level F1 averaged over K in PA%K.

    Recall is event-level: an event counts as detected if more than a K fraction of
    its points exceed the threshold (K=0 means at least one point).

    Precision is POINT-WISE (predicted points falling inside any true event, over all
    predicted points). Counting false positives as "predicted segments overlapping no
    event" — the natural-looking alternative — is degenerate: a single segment covering
    the whole series then has zero false positives and scores a perfect 1.0. See
    killtest/test_metrics.py, scenario `all_ones`."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    segs = segments(labels)
    if not segs:
        return np.nan
    out = []
    grid = _threshold_grid(scores, n_thresholds)
    for k in ks:
        best = 0.0
        for t in grid:
            pred = scores >= t
            npred = int(pred.sum())
            if npred == 0:
                continue
            tp = 0
            for s, e in segs:
                frac = pred[s:e + 1].mean()
                if (frac > 0) if k == 0.0 else (frac > k):
                    tp += 1
            rec = tp / len(segs)
            prec = float(np.logical_and(pred, labels == 1).sum()) / npred
            if prec + rec > 0:
                best = max(best, 2 * prec * rec / (prec + rec))
        out.append(best)
    return float(np.mean(out))


def all_metrics(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    return {
        "auc_pr": auc_pr(labels, scores),
        "vus_pr": vus_pr(labels, scores),
        "affil_f1": affiliation_f1(labels, scores),
        "event_f1": event_f1_pak(labels, scores),
    }
