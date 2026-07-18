param(
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $repoRoot "third_party\upstream-lock.json"
$upstreamRoot = Join-Path $repoRoot "third_party\upstream"
$entries = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json

foreach ($entry in $entries.repositories) {
    $destination = Join-Path $upstreamRoot $entry.name
    if (-not (Test-Path -LiteralPath $destination)) {
        if ($VerifyOnly) {
            throw "Missing upstream checkout: $($entry.name)"
        }
        git clone --filter=blob:none --no-checkout $entry.url $destination
    }

    $actualRemote = (git -C $destination remote get-url origin).Trim()
    if ($actualRemote -ne $entry.url) {
        throw "Remote mismatch for $($entry.name): expected $($entry.url), got $actualRemote"
    }

    $dirty = git -C $destination status --porcelain
    if ($dirty) {
        throw "Refusing to modify dirty upstream checkout: $destination"
    }

    if (-not $VerifyOnly) {
        git -C $destination fetch --depth 1 origin $entry.commit
        git -C $destination -c advice.detachedHead=false checkout --detach $entry.commit
        if ($entry.recursive_checkout) {
            git -C $destination submodule sync --recursive
            git -C $destination submodule update --init --recursive
        }
    }

    $actual = (git -C $destination rev-parse HEAD).Trim()
    if ($actual -ne $entry.commit) {
        throw "Commit mismatch for $($entry.name): expected $($entry.commit), got $actual"
    }

    foreach ($submodule in $entry.submodules) {
        $submodulePath = Join-Path $destination $submodule.path
        if (-not (Test-Path -LiteralPath $submodulePath)) {
            throw "Missing submodule $($submodule.name) for $($entry.name)"
        }
        $actualSubmodule = (git -C $submodulePath rev-parse HEAD).Trim()
        if ($actualSubmodule -ne $submodule.commit) {
            throw "Submodule mismatch for $($entry.name)/$($submodule.path): expected $($submodule.commit), got $actualSubmodule"
        }
    }
    Write-Host "$($entry.name) $actual"
}
