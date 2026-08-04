# Experimental Protocol (frozen 2026-08-01)

> **Correction issued 2026-08-03 (peer review F1).** The sentence below, "written **before** the
> final results are analysed", is true only in its narrow reading and has been used to support a
> broader claim that is false. A 230-row pilot (`results/killtest_results.csv`, 10:29; report
> `killtest_report.md`, 10:35) preceded this document on the same day and **already established the
> negative result** across six strata, with gaps of −0.257 to −0.433 and the explicit conclusion
> that untuned classical baselines beat zero-shot TSFM residual scoring comprehensively. The
> direction of the answer was therefore known when the interpretation rules below were written.
>
> What remains true: the rules were fixed before the *full-corpus* results existed, they were not
> relaxed afterwards, and the pilot's own stated gate was passed on a pre-existing criterion. What
> is not true, and must not be claimed in the paper, is that the rules were fixed before any data
> existed. See deviation D3 for a second, larger problem this correction surfaced.

This document fixes the full-experiment protocol. It is written **before** the final
results are analysed so that the analysis cannot drift toward a preferred conclusion.
Any deviation made after this point must be recorded in the "Deviations" section with
a reason and a date.

## 1. Research questions

- **RQ1** Cost-quality Pareto: where do zero-shot time-series foundation models (TSFMs)
  sit relative to tuned classical detectors on the quality × inference-cost plane?
- **RQ2** Break-even strata: does the TSFM−baseline quality gap depend on anomaly type,
  channel count, series length, or contamination rate?
- **RQ3** Operational rule: is there a stratum where deploying a TSFM is defensible, and
  what cost multiplier does it carry?

## 2. Data

| Dataset | Series | Channels | Source | Licence / access |
|---|---|---|---|---|
| UCR Anomaly Archive 2021 | 250 | 1 | cs.ucr.edu (direct zip) | public, no application |
| SMD (Server Machine Dataset) | 28 | 38 | OmniAnomaly GitHub | MIT, direct download |
| SMAP + MSL | 82 | 25 / 55 | TranAD GitHub mirror | public mirror; **NASA S3 origin now returns AccessDenied** |
| PSM (eBay) | 1 (40k-point prefix) | 25 | RANSynCoders GitHub | public, direct download |

**Reproducibility note worth reporting in the paper:** the canonical NASA telemanom S3
bucket (`s3-us-west-2.amazonaws.com/telemanom/data.zip`) is no longer publicly readable
as of 2026-08-01; SMAP/MSL had to be obtained from a third-party research mirror. Any
paper claiming SMAP/MSL reproducibility from the original source is now stale.

SMAP/MSL ships per-anomaly type labels (`point` = 52 channels, `contextual` = 26,
`mixed` = 4). This is the only anomaly-type stratification available without manual
annotation, so RQ2's anomaly-kind analysis is restricted to that subset.

## 3. Methods under test

### Zero-shot TSFMs (no fine-tuning, no adaptation)
| Name | Checkpoint | Params |
|---|---|---|
| Chronos-Bolt small | `amazon/chronos-bolt-small` | 48M |
| Chronos-Bolt base | `amazon/chronos-bolt-base` | 205M |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | 200M |
| TiRex | `NX-AI/TiRex` | 35M |

Two usage modes per backend, both derived from the *same* forward pass:
- **resid** — `|x_t − median forecast_t|`, divided by the 90th percentile of
  `|Δx|` over the training region (robust scale).
- **quantile** — interval-violation depth,
  `max(q10 − x, x − q90, 0) / (q90 − q10)`; a calibration-aware score that
  rewards a model for knowing when it is uncertain.

Protocol: rolling origin, context 512, horizon 64, non-overlapping forecast blocks
covering the whole test region. Multivariate series are scored per channel on the
12 highest-variance channels (cost cap), aggregated by the mean of the top quartile
of channel scores.

*Rationale for including two modes:* a single naive residual score would leave the
study open to the objection that TSFMs lost because they were used badly. The
quantile mode uses the models' native probabilistic output.

### Classical baselines
IsolationForest, LOF, k-NN distance (matrix-profile-like), and a small dense
autoencoder — all on sliding windows of the robust-normalised series.

Three tiers:
- **default** — one fixed sensible configuration (window 100).
- **tuned** — budgeted random search (6 configs) selecting on a *synthetic* validation
  task built from the training region only (anomalies injected at log-uniformly
  distributed lengths so the procedure is not biased to a particular window scale).
  **No test labels are used.** Search wall-clock is recorded and charged to the method
  in the cost analysis.
- **oracle** (optional runs) — best test-set configuration; an upper bound only, never
  presented as achievable.

## 4. Metrics

- **VUS-PR** (primary) — range-based AP with ramp-shaped label dilation, averaged over
  a buffer grid scaled to the mean anomaly length (Paparrizos et al., VLDB 2022).
- **AUC-PR** — point-wise, for continuity with older literature.
- **Affiliation F1** — reference implementation of Huet et al. (KDD 2022).
- **Event-level F1 with PA%K** — averaged over K ∈ {0, 0.2, 0.5, 0.8} (Kim et al., AAAI 2022).

Classic point-adjusted F1 is **not** reported as a headline number; post-PA metrics
remain fragile under adversarial stress testing (arXiv:2607.11969).

## 5. Cost measurement

Measured in a **fresh subprocess per (method, series, device)** so peak GPU memory is
not contaminated by other resident models. Reported quantities:
- **cold** wall-clock (includes weight loading) — the batch-job cost.
- **warm** wall-clock (model already resident) — the streaming/repeated-inference cost.
- throughput in points/second, peak GPU MB, process RSS MB.
- Both CUDA (RTX 4060 Laptop, 8 GB) and CPU-only.

The in-run `peak_mb` column of `full_results.csv` is **not** valid for memory
comparison (shared process, resident models); `cost_profile.csv` is authoritative.

The headline cost axis uses **warm throughput**, because charging every series a full
model-load is unrepresentative of deployment and would unfairly penalise TSFMs.

## 6. Analysis plan

1. Headline table: mean/median per detector-tier with bootstrap 95% CIs (2000 resamples).
2. Per-series gap: best TSFM minus best baseline, stratified by dataset, anomaly kind,
   channel count, length tercile, contamination tercile. Report mean gap, CI, and the
   fraction of series where any TSFM beats every baseline.
3. Pareto front on (warm throughput, VUS-PR); mark dominated methods.
4. Decision map: per stratum, the recommended family and the cost multiplier of the
   TSFM alternative.

## 7. Pre-committed interpretation rules

- A stratum is called a **TSFM-favourable regime** only if the mean gap is positive
  **and** the bootstrap CI excludes zero **and** n ≥ 10 series.
- If no such stratum exists, the paper reports a negative result with a decision map
  that says "do not deploy", and the contribution is the cost-aware map plus the
  error analysis — the framing does not get softened to hide this.
- Win-rate is reported alongside the mean gap in every stratum, so that a small number
  of TSFM wins cannot be silently averaged away, nor over-sold.

## Verification

### V1 — 2026-08-02: the quality results are independent of machine load

Deviation D2 established that throughput measurements are load-sensitive. The obvious next
question is whether the *quality* results are too, since the 5 504 records were collected
over many hours under wildly varying conditions: one to four concurrent workers, a period of
GPU-memory thrashing that slowed TiRex from 10 s to 217 s per series, differing
`KILLTEST_N_JOBS` caps, and analysis jobs running alongside.

They are not, and this was checked rather than assumed. `killtest/verify_determinism.py`
re-runs a stratified sample of (series, detector) combinations on an idle machine and
compares against the archive at two levels — the raw score arrays and the recomputed
VUS-PR. Across all four datasets and six detectors spanning both families:

| Quantity | Worst observed |
|---|---|
| Normalised score deviation, max over points | **0.00e+00** |
| Score rank correlation vs archive | **1.000000** |
| \|VUS-PR difference\| vs `full_results.csv` | **4.70e-06** |

The scores are bit-identical after float16 rounding and the metric differences are pure
floating-point noise. This is the expected result — every detector is seeded
(`IsolationForest(random_state=0)`, `torch.manual_seed(0)`, `default_rng(0)` for the
synthetic tuning anomalies) and the TSFM forward passes are deterministic quantile heads
with no sampling — but the archive was collected under conditions disorderly enough that
assuming it would have been negligent. **No re-run is warranted.**

Two load-sensitive columns *do* exist in `full_results.csv` — `runtime_s` and `tune_s` —
and they are not trustworthy for the same reason D2's numbers were not. They reach no
reported quantity: `figures.py` sources the Fig. 1 cost axis from `cost_profile.csv` for all
twelve detectors (the wall-clock fallback path is never taken), and the wall-clock column
printed by `analyze_full.pareto()` is a diagnostic that appears in no table or claim.

*Reproducibility dependency worth recording:* PSM is used as a **40 000-point prefix**
(`run_full.py --psm-len`, default 40 000). Calling `data.load_psm()` without that argument
returns all 87 841 points and will not reproduce the archived PSM rows.

### V2 — 2026-08-02: the tuned tier's 296/360 coverage is structural, not an unfinished run

The `tuned` tier exists for 296 of the 360 series. This was checked rather than assumed, because a
partially completed tier and a structurally bounded one have opposite implications for the paper.

`run_full.tune_baseline` carves its synthetic validation segment from the last 30 % of the training
region and returns `None` if that segment is shorter than 800 points; `process()` then writes no
row. A series therefore supports the tuned tier only if `n_train - int(0.7 * n_train) >= 800`, i.e.
`n_train >= 2664`. Checked against the true training length of all 360 series, the rule accounts
for the split **exactly, with zero exceptions**: every one of the 296 covered series has
`n_train >= 2690`, every one of the 64 uncovered has `n_train <= 2648`. The 64 are 40 SMAP/MSL
channels and 24 UCR series. Re-running `run_full --tiers tuned` over `nasa,ucr` produces no
additional rows, confirming there is nothing left to compute.

**Consequence for reporting.** The excluded series are short, and short series are easier in this
corpus. Default-tier VUS-PR on the 64 excluded versus the 296 covered: k-NN 0.432 / 0.298,
AE 0.367 / 0.295, LOF 0.344 / 0.255, IForest 0.230 / 0.125. A tuned-tier mean (n = 296) is
therefore **not comparable** with a default-tier mean (n = 360), and the line "k-NN tuned 0.309 <
default 0.322" in `experiment_report.md` §1 measures subset difficulty rather than the effect of
tuning. Paired on the 296 common series, tuning moves the baselines by at most ±0.03 VUS-PR and in
both directions:

| Detector | default (n=296) | tuned | Δ | 95 % CI | tuned wins |
|---|---|---|---|---|---|
| LOF | 0.255 | 0.283 | **+0.027** | [+0.009, +0.046] | 56 % |
| k-NN | 0.298 | 0.309 | +0.010 | [−0.004, +0.025] | 57 % |
| AE | 0.295 | 0.284 | −0.011 | [−0.027, +0.006] | 38 % |
| IForest | 0.125 | 0.110 | **−0.015** | [−0.028, −0.003] | 34 % |

All default-versus-tuned statements in the paper use the paired form. The headline comparison is
unaffected: it uses the default tier over all 360 series.

Note also that the per-series break-even gap uses the best baseline available on that series, which
on the 64 uncovered series means the default tier only. That direction is conservative — it can
only make the TSFMs look better — and they still lose in every stratum.

### V3 — 2026-08-02: the SMAP/MSL label file contains a duplicated channel with conflicting windows

`labeled_anomalies.csv` as distributed with the TranAD mirror has **82 rows over 81 distinct
channels**. `P-2` (SMAP) appears twice, with anomaly windows `[[5350, 6575]]` and `[[5300, 6420]]`
against an identical declared `num_values` of 8 209 and an identical class of `[point]`. The two
windows overlap but neither contains the other, so the choice changes the labelled anomaly mass by
about 9 % on that channel.

`data._nasa_meta` / `load_nasa` resolve the channel by first match, so P-2 was scored against
`[[5350, 6575]]`. The series count reported everywhere in this study is therefore **81**, not the
82 rows the file suggests, which reconciles the count in §2 of this protocol with the 360-series
total. Impact on aggregate results is negligible at one series in 360, but the defect is worth
reporting: two studies can publish different P-2 numbers with neither being wrong.

## Deviations

### D1 — 2026-08-02: event-level F1 and affiliation F1 recomputed after an adversarial metric audit

**What happened.** The first full run (360 series, archived under `results/v1_broken_metrics/`)
produced an event-level F1 in which zero-shot TSFMs appeared to *win decisively*
(TimesFM-quantile 0.830 vs kNN 0.478; median 1.000 on UCR), contradicting VUS-PR, AUC-PR
and affiliation F1. Rather than report the flattering number, the metric was audited with
degenerate detectors (`killtest/test_metrics.py`).

**Root cause.** Two independent defects, both in the *evaluation*, not the models:

1. *Event F1 with segment-counted false positives is saturable.* Counting a false positive
   as "a predicted segment overlapping no true event" means a single segment covering the
   entire series has zero false positives, giving precision = recall = 1. Quantile-mode TSFM
   scores are exactly zero on ~70 % of points, so the quantile threshold grid contained the
   value 0, at which `score >= 0` flags everything. Measured: `all_ones`, `all_zeros`, and a
   blob placed in the *wrong* location all scored a perfect 1.000.
2. *A constant score collapses any quantile threshold sweep onto "predict everything",*
   which also inflated affiliation F1.

**Fixes applied.**
- Event F1 now uses point-wise precision with event-level recall; the PA%K recall semantics
  are unchanged. Degenerate detectors now score 0.000 and a perfect detector 0.995.
- All best-over-threshold sweeps exclude thresholds flagging more than 30 % of points
  (`MAX_PREDICTED_RATE`), since no practitioner would deploy such an operating point.
- Anomaly scores are now persisted (`results/scores/*.npz`), so a future metric revision
  never again requires re-running the detectors.

**Residual, reported not fixed.** Affiliation F1 retains a high floor — a *random* detector
scores 0.669 and a wrongly-placed blob 0.546 on a long single-event series, because
affiliation credit is distance-based. This is a property of the published metric, not of
this implementation. Affiliation numbers are therefore always reported against the random
floor and never as an absolute quality level. VUS-PR similarly caps a point-perfect detector
at 0.784 because of ramp-label dilation; this is inherent to the definition and applies
identically to every method, so rankings are unaffected.

**Impact on conclusions.** VUS-PR (the pre-registered primary metric) and AUC-PR were
computed by unchanged code and are unaffected; the headline finding does not move. The full
experiment was nonetheless re-run so that all four metrics are valid on identical data.

### D2 — 2026-08-02: the entire cost profile was discarded and re-measured

**What happened.** The v1 cost profile (96 rows, archived as
`results/v1_contended_cost_profile.csv`) was found to be invalid on both device arms and
was regenerated from scratch. Neither defect was visible in the file itself — every row
carried `status=ok`.

**Root cause 1 — the CPU arm never ran on the CPU.** The child process selected CPU via
`os.environ["CUDA_VISIBLE_DEVICES"] = ""`. On this Windows / torch 2.6+cu124 build that
does *not* hide the GPU: `torch.cuda.is_available()` still returns `True` and allocation on
`cuda:0` still succeeds, while `torch.cuda.device_count()` drops to 0. So the 36 "cpu" rows
that succeeded were GPU runs — their `peak_mb` values are byte-identical to the
corresponding CUDA rows (ae 283.5 MB, chronos_base 761.6 MB, chronos_small 236.6 MB). The
12 rows that failed did so *because* their libraries consult `device_count()`: TimesFM
derives `global_batch_size = per_core_batch_size × device_count` and asserted on 0, and
TiRex deserialised its checkpoint onto `cuda:0`.

**Root cause 2 — every row, both devices, was measured under CPU contention.** The v1
profile was launched in the background while `analyze_full`, `analyze_wins` and `figures`
ran in the foreground. Re-measuring on an idle machine raises throughput for *every*
detector, by an uneven 2.1×–5.1×:

| Detector | v1 (contended) | idle re-measure | factor |
|---|---|---|---|
| ae | 3,418 | 17,275 | 5.1× |
| knn | 17,371 | 73,628 | 4.2× |
| chronos_small | 32,122 | 132,407 | 4.1× |
| lof | 11,531 | 43,236 | 3.7× |
| iforest | 6,929 | 25,890 | 3.7× |
| timesfm | 1,302 | 4,640 | 3.6× |
| tirex | 803 | 2,310 | 2.9× |
| chronos_base | 21,472 | 45,509 | 2.1× |

Because the distortion is uneven it is not a harmless constant factor: it **reverses at
least one ranking**. In v1, Chronos-Bolt base (21,472) appeared faster than k-NN (17,371);
on an idle machine k-NN (73,628) is clearly faster than Chronos-Bolt base (45,509). The
Chronos-Bolt small vs k-NN ordering, which the cost argument leans on, is preserved
(1.85× → 1.80×), but every absolute figure quoted from v1 is wrong.

**Fixes applied.**
- Device selection is now an explicit `KILLTEST_DEVICE` override read by `detectors.DEVICE`;
  `CUDA_VISIBLE_DEVICES` is no longer used. TimesFM has its module placement and
  `device_count` pinned; TiRex receives `device=` explicitly. GPU behaviour is unchanged
  (`device_count` was already 1 there), so the two arms are now genuinely comparable.
- `cost_profile.csv` gains an `other_py` column recording concurrent python processes per
  measurement. A healthy run reads 3 — the Windows venv `python.exe` is a launcher stub
  that re-execs the real interpreter, so the orchestrator and the measuring child each
  appear twice. Anything above that marks the row as suspect. This canary immediately
  caught two rows (`cuda/knn` and `cuda/ae` on `smd/machine-1-2`, `other_py=7`), which were
  deleted and re-measured.
- The re-run is strictly serial on an otherwise idle machine, and no analysis job may run
  while it is in progress.

**Impact on conclusions.** Only the cost axis is affected; no quality metric is touched, so
the negative headline result and every VUS-PR/AUC-PR/event-F1/affiliation number stand
unchanged. What must be regenerated from the new file: the throughput/memory table, the
cost axis of the Pareto figure (Fig. 1), and the "cost is not the bottleneck" argument's
supporting numbers. The qualitative claim behind that argument — that TSFM overhead is
concentrated in one-off model loading rather than inference, and that the accurate methods
are also the cheap ones — is unchanged in direction by the re-measurement.

### D3 — 2026-08-03: five components committed in the pilot report were never run, and were never logged

**What happened.** `killtest_report.md` §4, written 2026-08-01 as the gate for proceeding to the
full experiment, listed five items as *required* fixes before the full run, and `run_full.py`'s own
docstring names `stage1_rq_brief.md` / `killtest_report.md` as the pre-registration. None of the
five was carried out, and until this entry none was recorded as a deviation:

| Committed in the pilot report | Status |
|---|---|
| Replace the approximate VUS-PR with the official TSB-AD implementation | **not done** — the custom implementation in `killtest/metrics.py::range_pr_auc` was used throughout |
| Add an embedding-based usage mode (THEMIS-style) | **not done** — `detectors.py` still advertises an `embed` mode that is not implemented |
| Add a lightweight adaptation arm (linear probe / LoRA) | **not done** — out of scope by later decision, but that decision was never recorded |
| Add MOMENT as a fifth backend | **not done** |
| Measure energy via `nvidia-smi` sampling, and cost in $/M-points | **not done** — cost is reported as throughput and memory only |

**Why this matters more than the individual omissions.** The pilot report explicitly warned that two
usage modes would not be enough to retire the "用法太朴素" (the models were used too naively)
objection, and named embedding scoring as one of the additions needed. The paper nonetheless
presents residual + quantile as exactly that rebuttal. Worse, §6.3 nominates embedding-based
scoring as "the most informative thing we did not test" without disclosing that it was
pre-registered and dropped, and without noting that three published studies
(ChronosAD, THEMIS, and the adaptation study) report that the embedding route *works*.

**Consequence for the paper.** The scope of the negative result must be stated as what it is:
zero-shot **forecast-derived** scoring, in two variants, on four forecasting backends. It is not a
result about foundation models for anomaly detection in general, and the two-mode design does not
retire the naive-usage objection on its own. §4.2's framing and §6.5's limitations both need to say
so.

### D4 — 2026-08-03: the persisted score archive was stored in float16 and 22 % of it saturated

**What happened.** `run_full.save_scores` cast scores to `float16`, which saturates at 65 504.
k-NN and autoencoder distances on NASA and SMD, and TSFM quantile-mode scores, routinely exceed
that. 1 213 of 5 504 arrays (22.0 %) therefore contain non-finite values and 69 are entirely
non-finite, concentrated in NASA (all 81 channels affected for most detectors) and SMD.

**What it did and did not affect.** `full_results.csv` is unaffected: every metric there was
computed on the float64 scores before saving. Affected is everything recomputed *from the archive*,
which means the §6.3 mechanism analysis and the score-aggregation control. The damage is not random:
because NASA and SMD are hit hardest, the subset on which all detectors have usable arrays is
almost pure UCR, so an analysis that silently drops non-finite series becomes a univariate-only
analysis while appearing to cover the corpus.

**Why V1 did not catch it.** `verify_determinism.py` masks to finite entries before comparing
(`fin = np.isfinite(new16) & np.isfinite(stored)`), so the bit-for-bit reproduction it certifies is
silent exactly where the archive is broken. V1's conclusion about determinism stands; its implied
conclusion about archive integrity does not.

**Fix applied.** `save_scores` now writes `float32` (~420 MB for the full archive).
`repair_scores.py` recomputes and re-persists only the affected combinations, reading baseline
configurations back from the `config` column so the reproduction is exact rather than merely
equivalent, and checking each recomputed VUS-PR against the archived CSV value. Any mismatch is
reported rather than silently overwritten.
