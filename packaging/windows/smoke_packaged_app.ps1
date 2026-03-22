param(
    [string]$ExePath = ""
)

$ErrorActionPreference = "Stop"

if (-not $ExePath) {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $ExePath = Join-Path $RepoRoot "dist\Oscimorph\Oscimorph.exe"
}

if (-not (Test-Path $ExePath)) {
    throw "Packaged exe not found: $ExePath"
}

$env:OSCIMORPH_SKIP_STARTUP = "1"
$process = Start-Process -FilePath $ExePath -PassThru
Start-Sleep -Seconds 5

if ($process.HasExited) {
    throw "Packaged app exited early with code $($process.ExitCode)"
}

Stop-Process -Id $process.Id -Force

$appDataRoot = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "Oscimorph"
} else {
    Join-Path $HOME ".oscimorph"
}

$expected = @(
    (Join-Path $appDataRoot "debug"),
    (Join-Path $appDataRoot "output"),
    (Join-Path $appDataRoot "temp"),
    (Join-Path $appDataRoot "presets")
)

foreach ($path in $expected) {
    if (-not (Test-Path $path)) {
        throw "Expected runtime directory missing: $path"
    }
}

Write-Output "PACKAGED_SMOKE_OK"
