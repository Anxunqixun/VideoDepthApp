# -*- mode: python ; coding: utf-8 -*-
# Windows-oriented onefile spec.
# CANNOT be built on this Linux box: PyInstaller does not cross-compile.
# Build on a Windows PC with the same venv stack (Python 3.11, torch CPU) via:
#   pyinstaller VideoDepthApp-windows.spec
# The resulting VideoDepthApp.exe is the portable Windows binary the user asked for.
import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

SPECDIR = os.path.abspath(SPECPATH)
VENDOR = os.path.join(SPECDIR, "vendor")

datas = [
    (os.path.join(VENDOR, "checkpoints", "video_depth_anything_vits.pth"),
     os.path.join("vendor", "checkpoints")),
]
binaries = []
hiddenimports = [
    "torch", "torchvision", "cv2", "imageio", "imageio_ffmpeg",
    "matplotlib", "einops", "easydict", "tqdm", "numpy",
    "video_depth_anything", "video_depth_anything.video_depth",
    "utils", "utils.dc_utils", "utils.util",
    "tkinter",
]

for pkg in ("torch", "torchvision", "cv2", "imageio", "imageio_ffmpeg", "matplotlib"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:
        print("collect_all failed for", pkg, exc)

try:
    binaries += collect_dynamic_libs("torch")
except Exception:
    pass

block_cipher = None

a = Analysis(
    [os.path.join(SPECDIR, "app.py")],
    pathex=[SPECDIR, VENDOR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VideoDepthApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
