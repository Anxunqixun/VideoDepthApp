$ErrorActionPreference = "Stop"
$root = Join-Path (Get-Location) "dist\VideoDepthApp"
if (Test-Path $root) { Remove-Item -Recurse -Force $root }
New-Item -ItemType Directory -Force -Path $root | Out-Null

$prefix = python -c "import sys; print(sys.base_prefix)"
Write-Host "Copying Python from $prefix"
Copy-Item -Path (Join-Path $prefix "*") -Destination (Join-Path $root "python") -Recurse -Force

$py = Join-Path $root "python\python.exe"
$pyw = Join-Path $root "python\pythonw.exe"
if (-not (Test-Path $py)) { throw "python.exe missing in portable tree" }
if (-not (Test-Path $pyw)) { throw "pythonw.exe missing; tk GUI needs official Windows Python" }

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

Copy-Item vda_gui.py, vda_cli.py, engine.py $root
Copy-Item -Recurse models $root\models
Copy-Item "打开界面.bat","打开界面.vbs","命令行.bat" $root
@"
深度估计便携版（官方 Python，已签名）

双击「打开界面.bat」或「打开界面.vbs」启动窗口。
命令行：命令行.bat --image 图.jpg
         命令行.bat --video 片.mp4 --stride 2 --colormap turbo

不要只拷启动脚本，请使用整个 VideoDepthApp 文件夹。
离线可用，无需安装。
"@ | Set-Content -Encoding UTF8 (Join-Path $root "使用说明.txt")

Get-ChildItem $root -Recurse -File | Select-Object FullName, Length | Format-Table -AutoSize
$sig = Get-AuthenticodeSignature $py
Write-Host "python.exe signature:" $sig.Status $sig.SignerCertificate.Subject
