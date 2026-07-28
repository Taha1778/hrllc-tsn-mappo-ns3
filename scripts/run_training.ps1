param(
  [int]$Rounds = -1,
  [int]$Iterations = -1,
  [int]$K = -1,
  [int]$Seed = -1,
  [double]$Radius = -1.0,
  [double]$Power = -1.0,
  [double]$GammaDb = -9999.0,
  [switch]$ResetModel,
  [switch]$FreshStart,
  [int]$ValidationEvery = -1,
  [int]$ValidationSeeds = -1,
  [int]$CheckpointEvery = -1,
  [switch]$Preflight,
  [switch]$FinalEvaluateBest
)

$ErrorActionPreference = "Stop"

try {
  $ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
  $TorchPython = "C:\Users\tahar\AppData\Local\Programs\Python\Python312\python.exe"
  $BundledPython = "C:\Users\tahar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
  if (Test-Path -LiteralPath $TorchPython) {
    $Python = $TorchPython
  } elseif (Test-Path -LiteralPath $BundledPython) {
    $Python = $BundledPython
  } else {
    $Python = "python"
  }

  & $Python -c "import numpy, pandas, torch"
  if ($LASTEXITCODE -ne 0) {
    throw "Python dependencies are missing. Install them with: python -m pip install -r requirements.txt"
  }

  $Trainer = Join-Path $ProjectRoot "python\hrllc_tsn_trainer.py"
  $argsList = @($Trainer)
  if ($Rounds -ge 0) { $argsList += @("--rounds", $Rounds) }
  if ($Iterations -ge 0) { $argsList += @("--iterations", $Iterations) }
  if ($K -ge 0) { $argsList += @("--k", $K) }
  if ($Seed -ge 0) { $argsList += @("--seed", $Seed) }
  if ($Radius -ge 0) { $argsList += @("--radius", $Radius) }
  if ($Power -ge 0) { $argsList += @("--power", $Power) }
  if ($GammaDb -gt -9999.0) { $argsList += @("--gamma-db", $GammaDb) }
  if ($ResetModel) {
    $argsList += "--reset-model"
  }
  if ($FreshStart) {
    $argsList += "--fresh-start"
  }
  if ($Preflight) {
    $argsList += "--preflight"
  }
  if ($FinalEvaluateBest) {
    $argsList += "--final-evaluate-best"
  }
  if ($ValidationEvery -ge 0) { $argsList += @("--validation-every", $ValidationEvery) }
  if ($ValidationSeeds -ge 0) { $argsList += @("--validation-seeds", $ValidationSeeds) }
  if ($CheckpointEvery -ge 0) { $argsList += @("--checkpoint-every", $CheckpointEvery) }

  & $Python @argsList
  if ($LASTEXITCODE -ne 0) { throw "Training workflow failed." }
} catch {
  Write-Error $_
  exit 1
}
