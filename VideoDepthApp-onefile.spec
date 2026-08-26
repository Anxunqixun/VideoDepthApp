# -*- mode: python ; coding: utf-8 -*-
# Linux (or Windows, if rebuilt there) onefile. Large because of torch.
# PyInstaller cannot cross-compile: this spec on Linux produces a Linux binary.
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
    "matplotlib", "matplotlib.colormaps", "matplotlib.cm",
    "einops", "easydict", "tqdm", "decord", "numpy",
    "video_depth_anything", "video_depth_anything.video_depth",
    "video_depth_anything.dpt", "video_depth_anything.dpt_temporal",
    "video_depth_anything.dinov2", "video_depth_anything.dinov2_layers",
    "video_depth_anything.motion_module.motion_module",
    "video_depth_anything.motion_module.attention",
    "video_depth_anything.util.transform", "video_depth_anything.util.blocks",
    "utils", "utils.dc_utils", "utils.util",
    "tkinter", "tkinter.filedialog", "tkinter.messagebox", "tkinter.ttk",
]

for pkg in ("torch", "torchvision", "cv2", "imageio", "imageio_ffmpeg", "matplotlib", "decord"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:
        print("collect_all failed for", pkg, exc)


# uv CPython 3.11 is linked against Tcl/Tk 9 (not system 8.6)
_uvlib_candidates = [
    "/home/box/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/lib",
    "/home/box/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/lib",
]
for _uvlib in _uvlib_candidates:
    if os.path.isfile(os.path.join(_uvlib, "libtcl9.0.so")):
        binaries += [
            (os.path.join(_uvlib, "libtcl9.0.so"), "."),
            (os.path.join(_uvlib, "libtcl9tk9.0.so"), "."),
        ]
        if os.path.isdir(os.path.join(_uvlib, "tcl9.0")):
            datas += [(os.path.join(_uvlib, "tcl9.0"), "tcl9.0")]
        if os.path.isdir(os.path.join(_uvlib, "tk9.0")):
            datas += [(os.path.join(_uvlib, "tk9.0"), "tk9.0")]
        break

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
    excludes=["pytest", "IPython", "jupyter"],
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
    name="VideoDepthApp-onefile",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
