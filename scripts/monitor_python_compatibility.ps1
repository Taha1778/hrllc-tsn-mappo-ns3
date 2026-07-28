param(
  [double]$TargetFailureRate = 0.01
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$StatusPath = Join-Path $ProjectRoot "runs\workflow_status.json"
$MetricsPath = Join-Path $ProjectRoot "runs\final_test_metrics.json"

if (-not (Test-Path -LiteralPath $StatusPath)) {
  @{ state = "not_started" } | ConvertTo-Json -Compress
  exit 0
}

$status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
if ($status.status -eq "running") {
  @{ state = "running"; mode = $status.mode; action = "no_duplicate_started" } | ConvertTo-Json -Compress
  exit 0
}

if ($status.status -eq "completed" -and $status.mode -eq "training" -and -not (Test-Path -LiteralPath $MetricsPath)) {
  @{ state = "training_complete"; action = "final_evaluation_required" } | ConvertTo-Json -Compress
  exit 0
}

if (Test-Path -LiteralPath $MetricsPath) {
  $metrics = Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json
  $good = [double]$metrics.failure_rate -lt $TargetFailureRate
  @{ state = "final_metrics_ready"; failure_rate = [double]$metrics.failure_rate; good = $good } | ConvertTo-Json -Compress
  exit 0
}

@{ state = "idle"; workflow_status = $status.status; mode = $status.mode } | ConvertTo-Json -Compress
