param(
    [string]$Python = "py",
    [switch]$UseLauncher
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$tempRoot = Join-Path $repoRoot ".tmp"

function Invoke-VenvPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($UseLauncher) {
        & $Python -3.10 -m venv $venvRoot
    }
    else {
        & $Python -m venv $venvRoot
    }
}

$version = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the virtual-environment Python version"
}
if ($version -ne "3.10") {
    throw "The WCDT-compatible core environment requires Python 3.10, got $version"
}

New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-VenvPython -Arguments @(
    "-m",
    "pip",
    "install",
    "-r",
    (Join-Path $repoRoot "requirements\core-py310-windows.lock.txt")
)
Invoke-VenvPython -Arguments @(
    "-m",
    "pip",
    "install",
    "--no-deps",
    "-e",
    $repoRoot
)
Invoke-VenvPython -Arguments @(
    (Join-Path $repoRoot "scripts\verify_core_lock.py")
)
Invoke-VenvPython -Arguments @(
    "-m",
    "pytest",
    (Join-Path $repoRoot "tests"),
    "-q",
    "--basetemp",
    (Join-Path $tempRoot "pytest")
)
