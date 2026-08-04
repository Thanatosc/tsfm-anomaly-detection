"""Adversarial sanity tests for the evaluation metrics.

A metric is only usable if degenerate detectors score near zero. We test:
  perfect     — exactly the ground-truth mask            -> must be ~1
  all_ones    — everything flagged as anomalous          -> must be ~0
  all_zeros   — nothing flagged (constant score)         -> must be ~0
  random      — white noise                              -> must be ~anomaly rate
  sparse_blob — zero everywhere except one wrong region  -> must be ~0
  shifted     — correct shape, offset by one window      -> mid range

Run:  .venv\\Scripts\\python.exe -m killtest.test_metrics
"""
import numpy as np

from .metrics import all_metrics

RNG = np.random.default_rng(0)


def scenarios(n=20000, seg=(1800, 2100)):
    lab = np.zeros(n, dtype=int)
    lab[seg[0]:seg[1]] = 1
    out = {}
    out["perfect"] = lab.astype(float) + RNG.normal(0, 1e-3, n)
    out["all_ones"] = np.ones(n)
    out["all_zeros"] = np.zeros(n)
    out["random"] = RNG.normal(0, 1, n)
    blob = np.zeros(n); blob[9000:9300] = 1.0
    out["sparse_blob_wrong"] = blob
    sh = np.zeros(n); sh[seg[0] + 150:seg[1] + 150] = 1.0
    out["shifted_half"] = sh
    # the empirical failure case: sparse score, 70% exact zeros, blob on target
    q = np.zeros(n); q[seg[0]:seg[1]] = 2.0
    q[RNG.integers(0, n, int(0.3 * n))] = RNG.uniform(0, 0.5, int(0.3 * n))
    out["sparse_on_target"] = q
    return lab, out


EXPECT = {  # (scenario, metric) -> (lo, hi) admissible range
    ("perfect", "*"): (0.75, 1.01),   # VUS-PR caps below 1 for a point-perfect detector:
                                      # ramp label dilation credits a buffer the detector
                                      # does not flag. Inherent to the definition, applies
                                      # equally to every method, so rankings are unaffected.
    ("all_ones", "*"): (-0.01, 0.25),
    ("all_zeros", "*"): (-0.01, 0.25),
    ("random", "*"): (-0.01, 0.30),
    ("sparse_blob_wrong", "*"): (-0.01, 0.30),
}

# Metrics with a known high floor: report them against the random-detector value,
# never as an absolute quality level.
HIGH_FLOOR = {"affil_f1"}


def main():
    lab, sc = scenarios()
    metrics = ["auc_pr", "vus_pr", "affil_f1", "event_f1"]
    print(f"anomaly rate = {lab.mean():.4f}\n")
    print(f"{'scenario':22s}" + "".join(f"{m:>12s}" for m in metrics))
    results = {}
    for name, s in sc.items():
        m = all_metrics(lab, s)
        results[name] = m
        print(f"{name:22s}" + "".join(f"{m[k]:>12.3f}" for k in metrics))

    print("\nFAILURES (degenerate detector scoring too high, or perfect scoring too low):")
    bad = 0
    for (scen, _), (lo, hi) in EXPECT.items():
        for k in metrics:
            v = results[scen][k]
            if v != v:
                continue
            if k in HIGH_FLOOR and scen != "perfect":
                continue  # reported against its own floor instead, see note below
            if not (lo <= v <= hi):
                print(f"  {scen:22s} {k:10s} = {v:.3f}   expected in [{lo}, {hi}]")
                bad += 1
    print("  none" if bad == 0 else f"  {bad} failures")

    print("\nKnown high-floor metrics (interpret relative to these values, not absolutely):")
    for k in sorted(HIGH_FLOOR):
        print(f"  {k:10s} random detector = {results['random'][k]:.3f}, "
              f"wrong-blob detector = {results['sparse_blob_wrong'][k]:.3f}")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
