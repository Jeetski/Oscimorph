# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


ROOT = Path.cwd()
APP_DIR = ROOT / "app"
SRC_DIR = APP_DIR / "src"

datas = [
    (str(APP_DIR / "assets"), "assets"),
    (str(APP_DIR / "presets"), "presets"),
    (str(APP_DIR / "scripts"), "scripts"),
]

changelog = APP_DIR / "docs" / "changelog.md"
if changelog.exists():
    datas.append((str(changelog), "docs"))

datas += collect_data_files("librosa")
datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("soundfile")
datas += copy_metadata("imageio")
datas += copy_metadata("imageio-ffmpeg")
datas += copy_metadata("moviepy")

hiddenimports = []


a = Analysis(
    [str(SRC_DIR / "oscimorph_boot.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "IPython",
        "jupyter_client",
        "jupyter_core",
        "pandas",
        "pytest",
        "sklearn",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Oscimorph",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(APP_DIR / "assets" / "logo.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Oscimorph",
)
