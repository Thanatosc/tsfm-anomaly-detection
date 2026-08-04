# Progress checker for the cost profile re-run.
# Deliberately PowerShell-only: a python-based checker would spawn python processes and
# push the other_py contention canary above its baseline of 3, falsely flagging whichever
# row is being measured at that moment.
#
# Row count alone is a misleading progress signal here -- work per row varies ~100x. One
# TiRex/CPU row on an SMD machine is 341,748 effective points (~73 min) while a baseline
# row on a short UCR series is under a second. The ETA below is work-based, not row-based.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$csv = "$root\results\cost_profile.csv"
$total = 96   # 8 detectors x 6 series x 2 devices
$MAXCH = 12   # detectors.MAX_CHANNELS -- TSFM scoring is capped at 12 channels

# Fixed target sizes (n_test, raw channels) for the 6 profiled series.
$targets = @{
  "ucr/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620" = @(44795, 1)
  "ucr/250_UCR_Anomaly_weallwalk_2951_7290_7296"           = @(7517, 1)
  "smd/machine-1-1"                                        = @(28479, 38)
  "smd/machine-1-2"                                        = @(23694, 38)
  "nasa/P-1"                                               = @(8505, 25)
  "nasa/S-1"                                               = @(7331, 25)
}
$detectors = @("iforest", "lof", "knn", "ae", "chronos_small", "chronos_base", "timesfm", "tirex")

function Work($key) {
  $t = $targets[$key]
  return [double]$t[0] * [Math]::Min($t[1], $MAXCH)
}

if (-not (Test-Path $csv)) { Write-Output "no cost_profile.csv yet"; exit }
$rows = @(Import-Csv $csv)

Write-Output ("rows written : {0}/{1}" -f $rows.Count, $total)
foreach ($d in @("cuda", "cpu")) {
  $n = @($rows | Where-Object { $_.device -eq $d }).Count
  $bar = ("#" * [int]($n / 48 * 24)).PadRight(24, ".")
  Write-Output ("  {0,-5} {1,2}/48  [{2}]" -f $d, $n, $bar)
}

$bad = @($rows | Where-Object { $_.status -ne "ok" })
$dirty = @($rows | Where-Object { $_.other_py -and [int]$_.other_py -gt 3 })
Write-Output ("errors       : {0}" -f $bad.Count)
Write-Output ("contended    : {0}   (other_py > 3; re-measured at the end)" -f $dirty.Count)

# --- calibrate effective points/sec per detector from completed CPU rows ---
$rate = @{}
foreach ($d in $detectors) {
  $done = @($rows | Where-Object { $_.device -eq "cpu" -and $_.detector -eq $d -and $_.status -eq "ok" })
  $w = 0.0; $s = 0.0
  foreach ($r in $done) {
    $k = "$($r.dataset)/$($r.series)"
    if ($targets.ContainsKey($k)) { $w += (Work $k); $s += [double]$r.cold_s + [double]$r.runtime_s }
  }
  if ($s -gt 0) { $rate[$d] = $w / $s }
}

# --- in-flight measurement ---
# Row count can sit still for over an hour on a single TiRex/CPU row, so show how far
# through the current row we are rather than leaving it looking hung.
$kids = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -like "*--child*" })
$kidPids = $kids | ForEach-Object { $_.ProcessId }
$real = $kids | Where-Object { $kidPids -contains $_.ParentProcessId } | Select-Object -First 1
$inflight = $null
if ($real) {
  $cl = $real.CommandLine
  $ds = ([regex]::Match($cl, "--dataset (\S+)")).Groups[1].Value
  $se = ([regex]::Match($cl, "--series (\S+)")).Groups[1].Value
  $de = ([regex]::Match($cl, "--detector (\S+)")).Groups[1].Value
  $dv = ([regex]::Match($cl, "--device (\S+)")).Groups[1].Value
  $key = "$ds/$se"
  $inflight = @{ key = $key; det = $de; dev = $dv }
  $mins = ((Get-Date) - $real.CreationDate).TotalMinutes
  Write-Output ("in flight    : {0} {1} on {2}" -f $dv, $de, $key)
  if ($dv -eq "cpu" -and $rate.ContainsKey($de) -and $targets.ContainsKey($key)) {
    $expect = (Work $key) / $rate[$de] / 60.0
    $pct = [Math]::Min(100, $mins / $expect * 100)
    $bar = ("#" * [int]($pct / 100 * 20)).PadRight(20, ".")
    Write-Output ("               [{0}] {1:N0} of ~{2:N0} min ({3:N0}%)" -f $bar, $mins, $expect, $pct)
  } else {
    Write-Output ("               {0:N0} min elapsed" -f $mins)
  }
}

# --- work-based ETA for the CPU arm ---

$doneKeys = @{}
foreach ($r in $rows) { $doneKeys["$($r.device)|$($r.detector)|$($r.dataset)/$($r.series)"] = $true }

$eta = 0.0; $unknown = 0
foreach ($k in $targets.Keys) {
  foreach ($d in $detectors) {
    if ($doneKeys.ContainsKey("cpu|$d|$k")) { continue }
    if ($rate.ContainsKey($d)) { $eta += (Work $k) / $rate[$d] } else { $unknown++ }
  }
}
# credit the in-flight row for time already spent
if ($inflight -and $inflight.dev -eq "cpu" -and $rate.ContainsKey($inflight.det)) {
  $spent = ((Get-Date) - $real.CreationDate).TotalSeconds
  $eta = [Math]::Max(0, $eta - $spent)
}
$note = ""
if ($unknown -gt 0) { $note = "  (+{0} rows not yet calibrated)" -f $unknown }
Write-Output ("eta cpu arm  : {0:N1} h{1}" -f ($eta / 3600), $note)

Write-Output ("processes    : {0} handles (~{1} logical)" -f $kids.Count, [int]($kids.Count / 2))
Write-Output ""
$orch = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -like "*killtest*" })
if ($rows.Count -ge $total -and $orch.Count -eq 0) { Write-Output ">>> COST PROFILE FINISHED" }
elseif ($orch.Count -eq 0) { Write-Output ">>> STOPPED EARLY - rerun: powershell -ExecutionPolicy Bypass -File launch_cost.ps1  (done rows are skipped)" }
else { Write-Output ">>> still running" }
