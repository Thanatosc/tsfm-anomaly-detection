# Launch workers split by detector family, so that only ONE process holds the
# foundation models on the GPU. Three workers each loading all four TSFMs needs
# ~8.5 GB on an 8 GB card and thrashes (TiRex went from 10 s to 217 s per series).
#
# Safe to re-run: workers skip (series, detector, tier) combinations already present
# in ANY results\full_results*.csv, so no work is recomputed.
param([int]$BaselineWorkers = 2)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root '.venv\Scripts\python.exe'
$threads = [Math]::Max(2, [int]([Environment]::ProcessorCount / ($BaselineWorkers + 1)) - 1)

$env:HF_HUB_DISABLE_XET = '1'
$env:KILLTEST_N_JOBS = "$threads"
$env:KILLTEST_TORCH_THREADS = "$threads"
$env:OMP_NUM_THREADS = "$threads"
$env:MKL_NUM_THREADS = "$threads"

$common = @('-u', '-m', 'killtest.run_full', '--datasets', 'nasa,psm,smd,ucr', '--n-ucr', '250')

function Launch($name, $extra) {
    Start-Process -FilePath $py -ArgumentList ($common + $extra) -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root "results\$name.log") `
        -RedirectStandardError  (Join-Path $root "results\$name.err")
    Start-Sleep -Seconds 5
}

# GPU workers. TiRex alone costs about as much as the other three backends combined,
# so it gets its own process; splitting also keeps each process's resident weights small
# enough that the 8 GB card never thrashes (all four in three processes needs ~8.5 GB).
Launch 'gpu_a' @('--detectors', 'tirex_resid,tirex_quantile', '--tiers', 'default', '--tag', 'gpua')
Launch 'gpu_b' @('--detectors',
                 'chronos_small_resid,chronos_small_quantile,chronos_base_resid,chronos_base_quantile,timesfm_resid,timesfm_quantile',
                 '--tiers', 'default', '--tag', 'gpub')

# CPU workers: classical baselines, sharded across series
0..($BaselineWorkers - 1) | ForEach-Object {
    Launch "cpu$_" @('--detectors', 'iforest,lof,knn,ae', '--tiers', 'default,tuned',
                     '--shard', "$_/$BaselineWorkers", '--tag', "cpu$_")
}

Write-Host "launched 1 GPU worker + $BaselineWorkers CPU workers, $threads threads each"
Write-Host "check with: .venv\Scripts\python.exe check_progress.py"
