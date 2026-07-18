param(
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $repoRoot "third_party\upstream-lock.json"
$upstreamRoot = Join-Path $repoRoot "third_party\upstream"
$entries = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    $output = & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed (exit $LASTEXITCODE): git $($Arguments -join ' ')"
    }
    return $output
}

function Invoke-GitNetwork {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [int]$Attempts = 3
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            return Invoke-Git -Arguments $Arguments
        }
        catch {
            if ($attempt -eq $Attempts) {
                throw
            }
            Write-Warning "Git network attempt $attempt/$Attempts failed; retrying."
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

foreach ($entry in $entries.repositories) {
    $destination = Join-Path $upstreamRoot $entry.name
    if (-not (Test-Path -LiteralPath $destination)) {
        if ($VerifyOnly) {
            throw "Missing upstream checkout: $($entry.name)"
        }
        Invoke-GitNetwork -Arguments @(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            $entry.url,
            $destination
        )
    }

    $actualRemote = (
        Invoke-Git -Arguments @("-C", $destination, "remote", "get-url", "origin")
    ).Trim()
    if ($actualRemote -ne $entry.url) {
        throw "Remote mismatch for $($entry.name): expected $($entry.url), got $actualRemote"
    }

    $dirty = Invoke-Git -Arguments @(
        "-C", $destination, "status", "--porcelain"
    )
    if ($dirty) {
        throw "Refusing to modify dirty upstream checkout: $destination"
    }

    if (-not $VerifyOnly) {
        Invoke-GitNetwork -Arguments @(
            "-C", $destination, "fetch", "--depth", "1", "origin", $entry.commit
        )
        Invoke-Git -Arguments @(
            "-C",
            $destination,
            "-c",
            "advice.detachedHead=false",
            "checkout",
            "--detach",
            $entry.commit
        )
        if ($entry.recursive_checkout) {
            Invoke-Git -Arguments @(
                "-C", $destination, "submodule", "sync", "--recursive"
            )
            Invoke-GitNetwork -Arguments @(
                "-C",
                $destination,
                "submodule",
                "update",
                "--init",
                "--recursive"
            )
        }
    }

    $actual = (
        Invoke-Git -Arguments @("-C", $destination, "rev-parse", "HEAD")
    ).Trim()
    if ($actual -ne $entry.commit) {
        throw "Commit mismatch for $($entry.name): expected $($entry.commit), got $actual"
    }

    foreach ($submodule in $entry.submodules) {
        $submodulePath = Join-Path $destination $submodule.path
        if (-not (Test-Path -LiteralPath $submodulePath)) {
            throw "Missing submodule $($submodule.name) for $($entry.name)"
        }
        $actualSubmodule = (
            Invoke-Git -Arguments @("-C", $submodulePath, "rev-parse", "HEAD")
        ).Trim()
        if ($actualSubmodule -ne $submodule.commit) {
            throw "Submodule mismatch for $($entry.name)/$($submodule.path): expected $($submodule.commit), got $actualSubmodule"
        }
    }
    Write-Host "$($entry.name) $actual"
}
