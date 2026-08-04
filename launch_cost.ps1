# Launch exactly one detached cost-profiling orchestrator.
# Invoking Start-Process inline from Git Bash spawns the process twice (observed:
# two orchestrators with identical creation timestamps), which both duplicates rows
# and makes the two runs contend -- the very thing this profile must avoid.
# Calling this file with -File launches it once.
param([string]$Devices = "cuda,cpu", [int]$NPerDataset = 2)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "$root\.venv\Scripts\python.exe"
$env:HF_HUB_DISABLE_XET = "1"

Start-Process -FilePath $py `
  -ArgumentList "-u", "-m", "killtest.cost_profile", "--devices", $Devices, "--n-per-dataset", "$NPerDataset" `
  -RedirectStandardOutput "$root\results\cost.log" `
  -RedirectStandardError "$root\results\cost.err" `
  -WorkingDirectory $root -WindowStyle Hidden

Write-Output "launched cost_profile devices=$Devices n_per_dataset=$NPerDataset"
