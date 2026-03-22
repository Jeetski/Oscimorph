param(
    [switch]$SkipPyInstaller,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvPython = Join-Path $RepoRoot "app\.venv\Scripts\python.exe"
$PyInstallerSpec = Join-Path $RepoRoot "packaging\pyinstaller\oscimorph.spec"
$DistRoot = Join-Path $RepoRoot "dist"
$AppDist = Join-Path $DistRoot "Oscimorph"
$VendorDir = Join-Path $RepoRoot "app\vendor\ffmpeg"
$BundledFfmpeg = Join-Path $VendorDir "ffmpeg.exe"
$InstallerScript = Join-Path $RepoRoot "packaging\windows\oscimorph_installer.iss"

if (-not (Test-Path $VenvPython)) {
    throw "Missing virtualenv Python at $VenvPython. Run install_dependencies.bat first."
}

New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null

if (-not (Test-Path $BundledFfmpeg)) {
    & $VenvPython -c "import imageio_ffmpeg, pathlib, shutil; src = pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe()); dst = pathlib.Path(r'$BundledFfmpeg'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); print(dst)"
}

if (-not $SkipPyInstaller) {
    & $VenvPython -m pip install pyinstaller
    Remove-Item $AppDist -Recurse -Force -ErrorAction SilentlyContinue
    & $VenvPython -m PyInstaller --noconfirm --clean $PyInstallerSpec

    $VendorOut = Join-Path $AppDist "vendor\ffmpeg"
    New-Item -ItemType Directory -Force -Path $VendorOut | Out-Null
    Copy-Item $BundledFfmpeg (Join-Path $VendorOut "ffmpeg.exe") -Force
}

if (-not $SkipInstaller) {
    $Iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if (-not $Iscc) {
        $CandidateIscc = @(
            (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
            "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            "C:\Program Files\Inno Setup 6\ISCC.exe"
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($CandidateIscc) {
            $Iscc = @{ Source = $CandidateIscc }
        }
    }
    if (-not $Iscc) {
        Write-Warning "Inno Setup compiler (iscc.exe) not found. PyInstaller build completed, installer step skipped."
        exit 0
    }
    & $Iscc.Source $InstallerScript
}
