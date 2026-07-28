param(
  [ValidateSet("Preflight", "Full", "Final")]
  [string]$Mode = "Preflight",
  [switch]$FreshStart,
  [switch]$Background
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$Runner = Join-Path $PSScriptRoot "run_training.ps1"
$Runs = Join-Path $ProjectRoot "runs"
New-Item -ItemType Directory -Force -Path $Runs | Out-Null

if ($Mode -eq "Preflight") {
  & $Runner -Preflight
  exit $LASTEXITCODE
}

if ($Mode -eq "Final") {
  & $Runner -FinalEvaluateBest
  exit $LASTEXITCODE
}

# The trainer reads all campaign defaults from configs/default_config.json.
# Command-line flags below exist only when a caller intentionally overrides them.
$runnerArgs = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", $Runner
)
if ($FreshStart) { $runnerArgs += "-FreshStart" }

if ($Background) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $stdout = Join-Path $Runs "campaign_$stamp.out.log"
  $stderr = Join-Path $Runs "campaign_$stamp.err.log"
  $escapedRunner = $Runner.Replace("'", "''")
  $backgroundCommand = "& '$escapedRunner'"
  if ($FreshStart) { $backgroundCommand += " -FreshStart" }
  $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($backgroundCommand))
  $process = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-EncodedCommand", $encodedCommand) -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  Write-Host "Campaign started in the background."
  Write-Host "PID: $($process.Id)"
  Write-Host "Status: $Runs\workflow_status.json"
  Write-Host "Logs: $stdout and $stderr"
  exit 0
}

& powershell.exe @runnerArgs
exit $LASTEXITCODE
