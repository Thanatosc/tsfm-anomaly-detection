# Licence for the data

The MIT licence in `LICENSE` covers the *code* in this repository. It does not
cover the experimental output, which is released instead under

    Creative Commons Attribution 4.0 International (CC BY 4.0)
    https://creativecommons.org/licenses/by/4.0/

This applies to:

* `results/all_results.csv` and the shard files it is merged from
* `results/cost_profile.csv`, `results/correlation_levels.csv`,
  `results/smoothing.csv` and `results/v1_broken_metrics/`
* the archive of 5504 persisted anomaly-score arrays deposited on Zenodo
* the figures under `figures/`

You may share and adapt this material for any purpose, including commercially,
provided you give appropriate credit. Please cite the paper.

## What this repository does *not* license

The four benchmark corpora are not redistributed here and are not ours to
license. `killtest/data.py` fetches them from their original maintainers, whose
own terms apply:

| Corpus | Source | Terms |
|---|---|---|
| UCR Anomaly Archive 2021 | Wu and Keogh, UC Riverside | as stated by the archive |
| SMD | OmniAnomaly repository | MIT |
| SMAP / MSL | NASA, via the TranAD mirror | US Government public domain |
| PSM | eBay, RANSynCoders repository | as stated by the repository |

Pretrained model weights (Chronos, TimesFM, TiRex) are downloaded from their
publishers and carry their publishers' licences.
