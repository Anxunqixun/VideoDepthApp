# VideoDepthApp

离线 CPU 深度估计：用 ONNX Runtime 对图片和视频做单帧深度推理，运行时不需要 PyTorch。

本仓库是打包前的源码。推荐从源码安装运行；仓库里虽有打包脚本，但日常使用请以 `python vda_gui.py` / `python vda_cli.py` 为准。

## 功能

- **中文图形界面**（`vda_gui.py`）和 **命令行**（`vda_cli.py`），共用 `engine.py`。
- **图片**：单帧 T=1，走同一份 ONNX。
- **视频**：对抽取出的每一帧分别跑 ONNX，**没有**官方 Video-Depth-Anything 的 32 帧时序窗（优先体积和速度）。
- **抽帧间隔**（默认 2，仅视频）和可选的 **目标帧率**。
- **深度配色**：默认 `turbo`，可选 viridis / plasma / magma / inferno / jet / hot / ocean / cool / gray；可勾选或加 `--invert` 反转近远颜色。
- **CPU 占用百分比**（默认 30%）：`threads = max(1, round(核数 × 百分比 / 100))`，并应用到 ONNX Runtime（`intra_op_num_threads` / `inter_op_num_threads`）、`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `ORT_NUM_THREADS`，以及 OpenCV（`cv2.setNumThreads`）。百分比会被夹到 1–100。

推理侧固定行为（当前代码里没有对应开关）：长边超过 640 会先缩小（`max_res=640`），再 letterbox 到 518×518 送入网络；只使用 `CPUExecutionProvider`。

## 从源码安装与运行

`requirements.txt` 没有写死最低 Python 版本。运行时依赖是：

- `onnxruntime>=1.17`
- `opencv-python-headless>=4.8`
- `numpy>=1.24,<2`

Windows 便携打包的 GitHub Actions 使用 **Python 3.11**。从源码跑 GUI 需要带 **tkinter** 的 CPython（建议 3.10 或 3.11）。

```bash
git clone https://github.com/Anxunqixun/VideoDepthApp.git
cd VideoDepthApp
python3 -m pip install -r requirements.txt
```

启动界面：

```bash
python3 vda_gui.py
```

命令行示例：

```bash
# 图片 → 默认写到同目录 photo_depth.png
python3 vda_cli.py --image photo.jpg

# 指定输出、配色、反转、CPU 占用
python3 vda_cli.py --image photo.jpg -o depth.png --colormap turbo --cpu-percent 30

# 视频：每 2 帧取 1 帧（默认 stride=2）
python3 vda_cli.py --video clip.mp4 -o clip_depth.mp4

# 再按目标帧率抽帧，并用 magma 配色
python3 vda_cli.py --video clip.mp4 --stride 2 --fps 8 --colormap magma --cpu-percent 50
```

未同时给出 `--image` 或 `--video` 时，CLI 会打印帮助并以退出码 2 结束。两者都给时只处理图片。

仓库根目录的 `打开界面.bat` / `start-gui.bat` / `命令行.bat` 面向便携目录布局（同级的 `python\pythonw.exe`），**不是**从源码树直接双击用的启动方式。

## 命令行参数

`python3 vda_cli.py -h` 的内容如下（与 `vda_cli.py` 里 argparse 一致）：

```
usage: vda_cli.py [-h] [--image IMAGE] [--video VIDEO] [--output OUTPUT]
                  [--stride STRIDE] [--fps FPS]
                  [--colormap {turbo,viridis,plasma,magma,inferno,jet,hot,ocean,cool,gray}]
                  [--invert] [--cpu-percent N]

深度估计 CLI (ONNX CPU)

options:
  -h, --help            show this help message and exit
  --image IMAGE         处理单张图片并退出
  --video VIDEO         处理视频并退出
  --output OUTPUT, -o OUTPUT
                        输出文件路径
  --stride STRIDE, --frame-stride STRIDE
  --fps FPS, --target-fps FPS
  --colormap {turbo,viridis,plasma,magma,inferno,jet,hot,ocean,cool,gray}, -c {turbo,viridis,plasma,magma,inferno,jet,hot,ocean,cool,gray}
                        深度配色，默认 turbo
  --invert              反转近远颜色
  --cpu-percent N       CPU 占用百分比 1–100（默认 30，折算线程数至少为 1）
```

argparse 里 `--stride` / `--fps` 没有 help 文案。代码中的含义是：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--stride` / `--frame-stride` | `2` | 每 N 帧取 1 帧（至少为 1） |
| `--fps` / `--target-fps` | 不设 | 若为正，再按原视频 fps 折算：`综合间隔 = N × round(原fps / 目标fps)`；输出帧率 = `原fps / 综合间隔` |
| `-o` / `--output` | 输入同目录 | 图片默认 `{文件名}_depth.png`，视频默认 `{文件名}_depth.mp4` |

视频编码为 OpenCV `mp4v`。

## 图形界面

运行 `python3 vda_gui.py` 会打开标题为「深度估计」的窗口。选图片或视频，可选输出文件夹（空则用输入文件所在目录），再点「开始」。

图片对话框接受 png / jpg / jpeg / bmp / webp / tif / tiff；视频对话框接受 mp4 / avi / mov / mkv / webm / m4v。视频可改抽帧间隔 N（默认 2）和目标帧率（可空）。配色下拉默认「青橙 turbo（推荐）」，「反转近远（近处变暗）」对应 CLI 的 `--invert`。

「CPU 占用」是 1–100 的数字框和滑条，默认 30%，旁白会显示类似 `30% ≈ 2 线程 / 8 核`。开始前改这个值会按新的线程数重建 ONNX 会话。界面选项不会写盘，每次启动回到默认。

首次打开会在后台加载模型。图片完成后在窗口里预览深度图；视频会写出 mp4，并预览中间一帧。图片走 T=1；视频仍是逐帧 ONNX，没有 32 帧时序窗。

## 模型

仓库自带 `models/vda_vits_t1.onnx`：官方 **vits** 权重导出的 **T=1**、**INT8** 模型（`models/export_meta.txt` 记录 `kind=int8`、`net=518`、约 32 MB）。推理时在程序目录、脚本目录或当前工作目录查找 `models/vda_vits_t1.onnx`（或同名文件）。`models/inferno.npy` 仅在 OpenCV 配色表不可用时作为 inferno 回退。

日常使用直接用这份 ONNX 即可，运行时不需要 PyTorch。

若要重新导出，仓库里有真实的 `build_model.py`：从 Video-Depth-Anything 的 vits 权重导出固定 518×518 的 T=1 ONNX，再尝试 FP16 / 动态 INT8，并按与官方 `infer_image` 的相对误差挑选较小的一份，复制为 `models/vda_vits_t1.onnx`。该脚本需要 PyTorch、官方仓库源码和 `video_depth_anything_vits.pth`；路径目前写死为开发环境的 `/workspace/Video-Depth-Anything` 等，本机使用前必须改掉。它**不**在 `requirements.txt` 里。

## 仓库文件

| 路径 | 说明 |
| --- | --- |
| `vda_gui.py` | 中文 tkinter 界面 |
| `vda_cli.py` | 命令行入口 |
| `engine.py` | 共享 ONNX 推理、配色、读写 |
| `build_model.py` | 开发用：重新导出 ONNX（需 PyTorch 与上游权重） |
| `requirements.txt` | 运行时依赖 |
| `requirements-build.txt` | 运行时依赖 + Nuitka 打包依赖 |
| `models/vda_vits_t1.onnx` | INT8 T=1 vits |
| `models/export_meta.txt` | 导出元数据 |
| `models/inferno.npy` | inferno LUT 回退 |
| `scripts/build_windows_portable.ps1` | 组装 Windows 便携目录 |
| `scripts/build_nuitka.sh` | Nuitka onedir 打包 |
| `.github/workflows/build-windows.yml` | 在 Windows 上跑便携打包并上传 artifact |
| `打开界面.bat` / `start-gui.bat` / `打开界面.vbs` / `start-gui.vbs` / `命令行.bat` / `start-cli.bat` | 便携目录启动器 |

仓库里有打包脚本和 Actions，但**不要把未签名的 exe / `dist/` / Nuitka 产物当作主安装方式**。本 README 以从源码运行为准。

## 致谢

深度模型来自 [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) / Depth Anything 系列。本项目把它收成离线 CPU、ONNX Runtime、T=1 逐帧的小应用；视频路径没有使用官方 32 帧时序推理。
