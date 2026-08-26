# 深度估计 VideoDepthApp

Windows 请下 **便携文件夹**（官方签名的 pythonw.exe 启动，避免杀软误报），不要用旧的 Nuitka 单文件 exe。

解压后双击 `打开界面.bat`。


离线 CPU 深度估计。ONNX Runtime，无 PyTorch。两个程序：

- `vda-cli` 命令行：`--image` / `--video` / `--stride` / `--fps` / `--output`
- `vda-gui` 中文界面：选图片 / 选视频 / 抽帧间隔 / 目标帧率

图片走单帧 T=1。视频逐帧跑同一 ONNX（优先体积和速度，无官方 32 帧时序窗）。

Linux Nuitka 产物（本机已编）：

- onefile：`dist-onefile/vda-cli` ~96 MB，`dist-onefile/vda-gui` ~98 MB
- onedir（启动更快）：`dist/vda_cli.dist/`、`dist/vda_gui.dist/`

## 从源码运行

```bash
pip install -r requirements.txt
python vda_cli.py --image photo.jpg -o depth.png
python vda_gui.py
```

默认 `max_res=640`，网络输入 letterbox 到 518×518。

## Nuitka 打包

```bash
pip install -r requirements-build.txt
./scripts/build_nuitka.sh          # onedir
# Windows：GitHub Actions → Build Windows Nuitka → artifact VideoDepthApp-windows-nuitka
```

## 模型

`models/vda_vits_t1.onnx` 由官方 vits 权重导出 T=1 后 INT8 量化（约 32MB）。重新导出需要 PyTorch：`python build_model.py`
