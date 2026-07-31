param(
  [string]$Ns3Branch = "ns-3.44",
  [string]$CMakeVersion = "3.30.5"
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath([string]$Path) {
  $resolved = (Resolve-Path -LiteralPath $Path).Path
  $drive = $resolved.Substring(0, 1).ToLowerInvariant()
  $rest = $resolved.Substring(2).Replace("\", "/")
  return "/mnt/$drive$rest"
}

try {
  $ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
  $External = Join-Path $ProjectRoot "external"
  $Ns3Dir = Join-Path $External $Ns3Branch
  $ScratchSource = Join-Path $ProjectRoot "ns3\scratch\hrllc_tsn_mappo.cc"
  $ScratchTargetDir = Join-Path $Ns3Dir "scratch"
  $ScratchTarget = Join-Path $ScratchTargetDir "hrllc_tsn_mappo.cc"
  New-Item -ItemType Directory -Force -Path $External | Out-Null

  $cmakeName = "cmake-$CMakeVersion-linux-x86_64"
  $cmakeTar = Join-Path $External "$cmakeName.tar.gz"
  $cmakeDir = Join-Path $External $cmakeName
  $externalWsl = Convert-ToWslPath $External
  $cmakeDirWsl = "$externalWsl/$cmakeName"
  $cmakeTarWsl = "$externalWsl/$cmakeName.tar.gz"

  $hasLinuxCmake = $false
  wsl bash -lc "command -v cmake >/dev/null 2>&1"
  if ($LASTEXITCODE -eq 0) {
    $hasLinuxCmake = $true
  }

  if (-not $hasLinuxCmake -and -not (Test-Path -LiteralPath $cmakeDir)) {
    $url = "https://github.com/Kitware/CMake/releases/download/v$CMakeVersion/$cmakeName.tar.gz"
    Write-Host "Downloading local CMake $CMakeVersion..."
    wsl bash -lc "cd '$externalWsl' && wget -q -O '$cmakeTarWsl' '$url' && tar -xzf '$cmakeTarWsl'"
    if ($LASTEXITCODE -ne 0) { throw "Could not download or extract local CMake." }
  }

  $pathPrefix = ""
  if ($hasLinuxCmake) {
    $pathPrefix = ""
  } else {
    $pathPrefix = "export PATH='$cmakeDirWsl/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'; "
  }

  if (-not (Test-Path -LiteralPath $Ns3Dir)) {
    Write-Host "Cloning ns-3 branch $Ns3Branch..."
    wsl bash -lc "cd '$externalWsl' && git clone --depth 1 --branch '$Ns3Branch' https://gitlab.com/nsnam/ns-3-dev.git '$Ns3Branch'"
    if ($LASTEXITCODE -ne 0) { throw "Could not clone ns-3." }
  }

  New-Item -ItemType Directory -Force -Path $ScratchTargetDir | Out-Null
  Copy-Item -LiteralPath $ScratchSource -Destination $ScratchTarget -Force

  $ns3Wsl = Convert-ToWslPath $Ns3Dir
  Write-Host "Configuring ns-3..."
  wsl bash -lc "$pathPrefix cd '$ns3Wsl' && ./ns3 configure --enable-modules=core --disable-python --disable-tests --disable-examples"
  if ($LASTEXITCODE -ne 0) { throw "ns-3 configure failed." }

  Write-Host "Building ns-3 simulator..."
  wsl bash -lc "$pathPrefix cd '$ns3Wsl' && ./ns3 build hrllc_tsn_mappo"
  if ($LASTEXITCODE -ne 0) { throw "ns-3 build failed." }

  Write-Host "ns-3 setup complete."
  Write-Host "Simulator target: $ScratchTarget"
} catch {
  Write-Error $_
  exit 1
}
