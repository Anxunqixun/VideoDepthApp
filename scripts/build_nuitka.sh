#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
COMMON=(
  --standalone
  --assume-yes-for-downloads
  --jobs="${JOBS:-8}"
  --output-dir=dist
  --include-data-files=models/vda_vits_t1.onnx=models/vda_vits_t1.onnx
  --include-data-files=models/inferno.npy=models/inferno.npy
  --include-package=onnxruntime
  --include-package-data=onnxruntime
  --nofollow-import-to=IPython
  --nofollow-import-to=pytest
  --nofollow-import-to=unittest
  --nofollow-import-to=setuptools
  --nofollow-import-to=pip
  --nofollow-import-to=numpy.tests
  --nofollow-import-to=onnxruntime.datasets
  --nofollow-import-to=onnxruntime.tools
  --nofollow-import-to=onnxruntime.quantization
  --remove-output
)
echo "Building vda-cli..."
"$PY" -m nuitka "${COMMON[@]}" --output-filename=vda-cli --nofollow-import-to=tkinter vda_cli.py
echo "Building vda-gui..."
"$PY" -m nuitka "${COMMON[@]}" --output-filename=vda-gui --enable-plugin=tk-inter vda_gui.py
echo "Done. Folders: dist/vda_cli.dist/ and dist/vda_gui.dist/"
du -sh dist/vda_cli.dist dist/vda_gui.dist || true
