param(
  [switch]$Background
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$Python = "C:\Users\tahar\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
$Suite = Join-Path $ProjectRoot "python\paper_reference_full_suite.py"
$Output = Join-Path $ProjectRoot "runs\run03_paper_reference_suite"
$Status = Join-Path $ProjectRoot "runs\run03_reference_workflow_status.json"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$statusJson = @{ status = "running"; mode = "full_source_python_reference"; started_utc = (Get-Date).ToUniversalTime().ToString("o") } | ConvertTo-Json
Set-Content -LiteralPath $Status -Value $statusJson -Encoding utf8

$command = "& '$Python' '$Suite' --output-dir '$Output'; if (`$LASTEXITCODE -eq 0) { @{ status = 'completed'; mode = 'full_source_python_reference'; completed_utc = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath '$Status' -Encoding utf8 } else { @{ status = 'failed'; mode = 'full_source_python_reference'; exit_code = `$LASTEXITCODE } | ConvertTo-Json | Set-Content -LiteralPath '$Status' -Encoding utf8; exit `$LASTEXITCODE }"

if ($Background) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $outLog = Join-Path $Output "run03_$stamp.out.log"
  $errLog = Join-Path $Output "run03_$stamp.err.log"
  $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
  $process = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-EncodedCommand", $encoded) -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  Write-Host "Run 3 started in the background. PID: $($process.Id)"
  Write-Host "Status: $Status"
  exit 0
}

& powershell.exe -NoProfile -Command $command
exit $LASTEXITCODE
