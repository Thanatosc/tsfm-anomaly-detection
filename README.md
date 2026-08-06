# When Do Zero-Shot Time-Series Foundation Models Pay Off for Anomaly Detection?

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21789152.svg)](https://doi.org/10.5281/zenodo.21789152)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21789216.svg)](https://doi.org/10.5281/zenodo.21789216)

Code, results and analysis for the paper *"When Do Zero-Shot Time-Series
Foundation Models Pay Off for Anomaly Detection? Cost Is Not the Barrier, and
the Bottleneck Is the Score Conversion."*

Time-series foundation models are increasingly proposed for anomaly detection on
two assumptions: that pretrained forecasting skill transfers to detection, and
that inference cost is what stands in the way of deploying them. We tested both
on 360 series from four public benchmarks, comparing eight zero-shot
configurations of four pretrained forecasters against four classical detectors
under a three-tier protocol, for 5504 result records and a separately measured
cost profile.

Neither assumption survived. Averaged over all series the best zero-shot
configuration reaches **0.080 VUS-PR against 0.322** for a k-nearest-neighbour
detector left at its defaults — a quarter of a baseline that was given no
per-series adaptation — and its best per-series win rate is 19 %. Cost does not
explain that and inverts it: once weights are resident the fastest foundation
model beats the strongest baseline's throughput by **1.78×**, so these models
occupy the cost–quality frontier only at its cheap and inaccurate corner.

Two results locate the deficit. Within a series, *lower* forecast error goes with
*better* detection (rank correlation **−0.415**), which contradicts the published
explanation that these models fail by reconstructing anomalies too faithfully.
And a score-level moving average — no labels, no extra inference, applied equally
to every detector — lifts them from 25 % to **53 %** of the best classical
detector while leaving that detector unmoved. The bottleneck is the conversion
from forecast to anomaly score, not the forecast.

A fourth result is methodological, and is the one we would most like reused: of
the four metrics we used, **one assigns a perfect score to detectors that flag
everything, flag nothing, or predict in the wrong place**. Before we caught it,
our own first run reported all four quantile-mode configurations beating every
baseline. `killtest/test_metrics.py` is the audit that caught it.

## What is in this repository

| Path | Contents |
|---|---|
| `killtest/` | the experimental harness: loaders, detectors, metrics, runner, cost profiler, determinism check, figure generation |
| `results/all_results.csv` | the 5504 result records, merged and deduplicated — the file every table in the paper is computed from |
| `results/full_results_*.csv` | the raw per-worker shards `all_results.csv` is merged from |
| `results/cost_profile.csv` | the isolated cost profile (cold/warm wall-clock, throughput, peak GPU memory, RSS) |
| `results/v1_broken_metrics/` | the discarded first run, kept deliberately: it is what the failing metric looked like before the audit |
| `results/correlation_levels.csv`, `results/mechanism_v2.csv`, `results/smoothing.csv` | the mechanism analysis and the score-smoothing control, already computed — Figure 5 rebuilds from these without the score archive |
| `figures/` | the five figures, twice: `.pdf` is the vector artwork the manuscript uses, `.png` is a 200 dpi raster so GitHub can render them inline |
| `protocol.md` | the pre-registered protocol, and the deviation log recording every departure from it |

## What is not, and where it lives

Three things are deliberately absent.

**The persisted anomaly scores** — 5504 `.npz` arrays, 234 MB — are deposited on
Zenodo at **[10.5281/zenodo.21789216](https://doi.org/10.5281/zenodo.21789216)**.
They let you recompute every metric in the paper, including metrics we did not
report, without running a single model. The record holds ten parts,
`scores_part01.zip` through `scores_part10.zip`; extract every one into
`results/` and they merge into a single `results/scores/`. `SHA256SUMS.txt`
verifies them, and `make_score_parts.py` here is the script that produced the
split, so you can check that the parts reconstitute exactly what we had.

**The four benchmark corpora** are not redistributed here; they are not ours to
license, and their maintainers already version them. See
[Getting the data](#getting-the-data).

**The manuscript itself.** This repository is the code and data artefact, not the
paper. The paper is under review at a subscription journal, where the accepted
version may only be self-archived after an embargo, so keeping the prose out
avoids both that constraint and any question of whether the MIT licence below
reaches text it was never meant to cover. Everything the paper asserts about
detector quality, cost and metric behaviour is checkable from what is here.

## Reproducing

There are three levels, in increasing order of cost. Most readers want level 1 or 2.

### 0. Install

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
```

Python 3.13.7. `requirements.txt` pins direct dependencies;
`requirements-lock.txt` is the full transitive closure actually installed. One
dependency (`affiliation`) is pinned to a git commit rather than a release
because it is not on PyPI and its API has changed without version bumps.

`torch` is pinned to a CUDA 12.4 build. Everything reproduces on CPU, only more
slowly — drop the `+cu124` suffix and the extra index URL.

### 1. Figures, tables and the metric audit — minutes, no GPU, no downloads

Everything the paper claims about detector quality, cost and metric behaviour is
reproducible from the CSVs already in this repository:

```bash
python -m killtest.figures          # Figures 1-4
python make_fig5.py                 # Figure 5, from results/mechanism_v2.csv and smoothing.csv
python -m killtest.analyze_full     # the stratified tables
python -m killtest.test_metrics     # the adversarial metric audit
```

Two further checks answer objections the paper raises against itself, and both are
released rather than asserted:

```bash
python -m killtest.horizon_check          # writes results/horizon_detrend.csv
pip install vus                           # not a harness dependency; install to run the next one
python -m killtest.vus_reference_check    # writes results/vus_reference_check.csv
```

`horizon_check` measures the period-64 ramp that non-overlapping forecast blocks
inject into every foundation-model score, and asks what removing it is worth: on
182 series the ramp accounts for 2 % of what the score-level moving average
recovers, so the smoothing result of Section 5.6 is not an artefact of our own
block schedule. `vus_reference_check` compares our VUS-PR against the reference
implementation of Paparrizos et al. on 1 368 paired values spanning all four
benchmarks; they agree at Spearman 0.991, and the headline ratio moves from 34.0 %
to 34.1 %. Install `vus` in a throwaway environment — it pins old numpy,
tensorflow and numba, and will fight the harness's own pins.

`test_metrics` needs no data at all: it synthesises the seven constructed
detectors (the ground truth itself, flag everything, flag nothing, white noise, a
contiguous blob placed at the true anomaly, the same blob placed elsewhere, and a
prediction displaced by half a window) and reports what each metric assigns
them. That is the whole audit — you should see a random detector scored 0.669 and
a half-displaced prediction scored 0.992 by the metric we single out.

> **A quirk worth knowing.** `killtest.analyze_full.load()` treats the path
> `results/full_results.csv` as a *virtual* name: when it sees that filename it
> globs `results/full_results*.csv` and deduplicates on
> `(dataset, series, detector, tier)`. There is no file by that name, and that is
> not an error. `results/all_results.csv` is the already-merged export of exactly
> those 5504 rows, provided for anyone who would rather not reimplement the merge.

### 2. Recomputing from the persisted scores — needs the Zenodo archive

Download all ten parts, extract each into `results/`, then:

```bash
python analyze_mechanism_v2.py      # regenerates mechanism_v2.csv, correlation_levels.csv, smoothing.csv
python check_smoothing.py           # the score-smoothing control on its own
```

Score files are named `{dataset}__{series}__{detector}__{tier}.npz` with the
score vector under the key `scores`. This is the level at which you can
substitute your own metric, or your own score conversion, and see what it says
about the same detectors on the same series — which is the reuse we are most
hoping for, given that the paper's finding is about the conversion step.

### 3. Full re-run — GPU, and the benchmark data

```bash
python -m killtest.run_full --datasets ucr,smd,nasa,psm --tiers default,tuned
```

Shard across workers with `--shard i/n` (process only items where
`index % n == i`); the shipped `full_results_*.csv` files are the output of one
such split. `--n-ucr 0` means all 250 UCR series. Measured on an RTX 4060 Laptop
with 8 GB and the same machine's 24-thread CPU.

The cost profile is deliberately a separate run, one fresh subprocess per
(method, series, device) triple, so peak GPU memory reflects one resident model:

```bash
python -m killtest.cost_profile --devices cuda,cpu
```

## Getting the data

`killtest/data.py` reads from a local `data/` tree and does not download. Fetch
each corpus from its own maintainer and unpack so the paths below exist:

```
data/UCR/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/*.txt
data/SMD/OmniAnomaly-master/ServerMachineDataset/{train,test,test_label}/*.txt
data/NASA/TranAD-main/data/SMAP_MSL/{train,test}/*.npy   + labeled_anomalies.csv
data/PSM/{train.csv,test.csv,test_label.csv}
```

| Corpus | Where |
|---|---|
| UCR Anomaly Archive 2021 | Wu and Keogh, UC Riverside |
| SMD | the OmniAnomaly repository |
| SMAP / MSL | the TranAD mirror. The canonical NASA S3 bucket returned `AccessDenied` throughout data collection; this is recorded in §4.1 of the paper |
| PSM | the RANSynCoders repository |

## Determinism

```bash
python -m killtest.verify_determinism --n 2
```

Re-runs a sample of (dataset, detector) pairs and checks the scores match. Every
number in the paper was re-derived from the repaired archive after a deviation
logged in `protocol.md`; the deviation log there is the honest record of what
went wrong, including the two cost-measurement failures that cost us an entire
profile.

## Citing

The paper is under review. Until it appears, cite the archived artefacts:

| | DOI |
|---|---|
| Code (this repository) | [10.5281/zenodo.21789152](https://doi.org/10.5281/zenodo.21789152) — concept DOI, always resolves to the latest version |
| Data (persisted scores) | [10.5281/zenodo.21789216](https://doi.org/10.5281/zenodo.21789216) |

This section will be updated with the article reference on acceptance.

## Licence

Code is MIT (`LICENSE`). Experimental output — the result records, the cost
profile, the score archive and the figures — is CC BY 4.0 (`LICENSE-DATA.md`).
The benchmark corpora and the pretrained weights carry their own licences and are
not covered by either.
