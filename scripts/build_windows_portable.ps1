$ErrorActionPreference = "Stop"

$root = Join-Path (Get-Location) "dist\VideoDepthApp"
if (Test-Path $root) { Remove-Item -Recurse -Force $root }
New-Item -ItemType Directory -Force -Path $root | Out-Null

$pydir = Join-Path $root "python"
New-Item -ItemType Directory -Force -Path $pydir | Out-Null

$prefix = & python -c "import sys; print(sys.base_prefix)"
Write-Host "Copying Python from $prefix"
Copy-Item -Path (Join-Path $prefix "*") -Destination $pydir -Recurse -Force

$py = Join-Path $pydir "python.exe"
$pyw = Join-Path $pydir "pythonw.exe"
if (-not (Test-Path $py)) { throw "python.exe missing in portable tree" }
if (-not (Test-Path $pyw)) { throw "pythonw.exe missing; tk GUI needs official Windows Python" }

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

Copy-Item -Path "vda_gui.py","vda_cli.py","engine.py" -Destination $root
Copy-Item -Path "models" -Destination (Join-Path $root "models") -Recurse
Get-ChildItem -File | Where-Object { $_.Extension -in ".bat",".vbs",".txt" } | Copy-Item -Destination $root

$readme = Join-Path $root "readme.txt"
Set-Content -Path $readme -Encoding UTF8 -Value @(
    "Portable VideoDepthApp",
    "Double-click start-gui.bat (or the Chinese launcher if present).",
    "CLI: start-cli.bat --image photo.jpg",
    "Keep the whole folder. Offline, no install."
)

Get-ChildItem $root -Recurse -File | Select-Object FullName, Length | Format-Table -AutoSize
$sig = Get-AuthenticodeSignature -FilePath $py
Write-Host ("python.exe signature: {0}" -f $sig.Status)
