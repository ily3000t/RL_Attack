param(
    [string]$Python = "py",
    [switch]$UseLauncher
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($UseLauncher) {
        & $Python -3.10 -m venv $venvRoot
    }
    else {
        & $Python -m venv $venvRoot
    }
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $repoRoot "requirements\wcdt-compat.txt")
& $venvPython -m pip install --no-deps -e $repoRoot
& $venvPython -m pytest (Join-Path $repoRoot "tests") -q
