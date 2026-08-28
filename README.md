# VideoDepthApp

离线深度估计：用 ONNX Runtime 对图片和视频做单帧深度推理，运行时不需要 PyTorch。默认是 **CPU**；源码里可以选装 **NVIDIA CUDA GPU**（`onnxruntime-gpu`）。

**支持的使用方式是从源码运行。** 克隆后 `pip install` 再 `python vda_gui.py` / `python vda_cli.py` 即可。仓库里虽有便携目录 / Nuitka 脚本，只作参考，不是日常安装路径。

## 功能

- **中文图形界面**（`vda_gui.py`）和 **命令行**（`vda_cli.py`），共用 `engine.py`。
- **图片**：单帧 T=1，走同一份 ONNX。
- **视频**：对抽取出的每一帧分别跑 ONNX，**没有**官方 Video-Depth-Anything 的 32 帧时序窗（优先体积和速度）。
- **抽帧间隔**（默认 2，仅视频）和可选的 **目标帧率**。
- **深度配色**：默认 `turbo`，可选 viridis / plasma / magma / inferno / jet / hot / ocean / cool / gray；可勾选或加 `--invert` 反转近远颜色。
- **CPU 占用百分比**（默认 30%）：`threads = max(1, round(核数 × 百分比 / 100))`，并应用到 ONNX Runtime（`intra_op_num_threads` / `inter_op_num_threads`）、`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `ORT_NUM_THREADS`，以及 OpenCV（`cv2.setNumThreads`）。百分比会被夹到 1–100。**只约束 CPU 线程**；勾选 GPU 且 CUDA 生效时，不会拿它去限制显卡占用（设置仍留给 CPU 回退路径用）。
- **可选 GPU**：界面勾选「使用 GPU」，或命令行加 `--gpu`。会话按 `CUDAExecutionProvider` → `CPUExecutionProvider` 创建；CUDA 不可用时回退 CPU，并给出中文提示，不会崩溃。GPU 路径只有 NVIDIA CUDA，没有 DirectML / TensorRT。

推理侧固定行为（当前代码里没有对应开关）：长边超过 640 会先缩小（`max_res=640`），再 letterbox 到 518×518 送入网络。

## 从源码安装与运行

运行时依赖没有写死最低 Python 版本。从源码跑 GUI 需要带 **tkinter** 的 CPython（建议 3.10 或 3.11）。

`onnxruntime` 与 `onnxruntime-gpu` **不能同时安装**。CPU 与 GPU 用两份 requirements，opencv / numpy 的版本范围相同。

**CPU（默认）：**

```bash
git clone https://github.com/Anxunqixun/VideoDepthApp.git
cd VideoDepthApp
python3 -m pip install -r requirements.txt
```

`requirements.txt` 是：`onnxruntime>=1.17`、`opencv-python-headless>=4.8`、`numpy>=1.24,<2`。

**GPU（可选，NVIDIA CUDA）：**

先卸掉 CPU 包，再装 GPU 包（或在干净环境里只装 GPU 那份）：

```bash
python3 -m pip uninstall -y onnxruntime
python3 -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` 把 `onnxruntime` 换成 `onnxruntime-gpu>=1.17`，其余与 CPU 相同。源码 GPU 安装只替换这一个 pip 包，不附带完整 CUDA toolkit。本机还需要与 wheel 匹配的 NVIDIA 驱动 / CUDA。装好后仍要在界面勾选「使用 GPU」或给 CLI 加 `--gpu`。

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

# 优先 CUDA；不可用则回退 CPU，并打印实际 provider
python3 vda_cli.py --image photo.jpg --gpu

# 视频：每 2 帧取 1 帧（默认 stride=2）
python3 vda_cli.py --video clip.mp4 -o clip_depth.mp4

# 再按目标帧率抽帧，并用 magma 配色
python3 vda_cli.py --video clip.mp4 --stride 2 --fps 8 --colormap magma --cpu-percent 50
```

未同时给出 `--image` 或 `--video` 时，CLI 会打印帮助并以退出码 2 结束。两者都给时只处理图片。

## 命令行参数

`python3 vda_cli.py -h` 的内容如下（与 `vda_cli.py` 里 argparse 一致）：

```
usage: vda_cli.py [-h] [--image IMAGE] [--video VIDEO] [--output OUTPUT]
                  [--stride STRIDE] [--fps FPS]
                  [--colormap {turbo,viridis,plasma,magma,inferno,jet,hot,ocean,cool,gray}]
                  [--invert] [--cpu-percent N] [--gpu]

深度估计 CLI (ONNX，可选 CUDA GPU)

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
  --cpu-percent N       CPU 占用百分比 1–100（默认 30，折算线程数至少为 1；不限制 GPU）
  --gpu                 优先使用 NVIDIA CUDA；不可用则回退 CPU，并打印实际 provider
```

argparse 里 `--stride` / `--fps` 没有 help 文案。代码中的含义是：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--stride` / `--frame-stride` | `2` | 每 N 帧取 1 帧（至少为 1） |
| `--fps` / `--target-fps` | 不设 | 若为正，再按原视频 fps 折算：`综合间隔 = N × round(原fps / 目标fps)`；输出帧率 = `原fps / 综合间隔` |
| `-o` / `--output` | 输入同目录 | 图片默认 `{文件名}_depth.png`，视频默认 `{文件名}_depth.mp4` |
| `--gpu` | 关 | 优先 `CUDAExecutionProvider`，失败则 `CPUExecutionProvider` |

视频编码为 OpenCV `mp4v`。

## 图形界面

运行 `python3 vda_gui.py` 会打开标题为「深度估计」的窗口。选图片或视频，可选输出文件夹（空则用输入文件所在目录），再点「开始」。

图片对话框接受 png / jpg / jpeg / bmp / webp / tif / tiff；视频对话框接受 mp4 / avi / mov / mkv / webm / m4v。视频可改抽帧间隔 N（默认 2）和目标帧率（可空）。配色下拉默认「青橙 turbo（推荐）」，「反转近远（近处变暗）」对应 CLI 的 `--invert`。

「CPU 占用」是 1–100 的数字框和滑条，默认 30%，旁白会显示类似 `30% ≈ 2 线程 / 8 核`。「使用 GPU」勾选后按 CUDA → CPU 建会话；没装 `onnxruntime-gpu` 或本机没有可用 CUDA 时，状态栏会用中文说明已回退 CPU，程序不会崩。改 GPU 勾选或 CPU 占用后，会按现有锁/会话模式重建 ONNX 会话。界面选项不会写盘，每次启动回到默认。

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
| `requirements.txt` | CPU 运行时依赖（`onnxruntime`） |
| `requirements-gpu.txt` | GPU 运行时依赖（`onnxruntime-gpu`，其余相同） |
| `requirements-build.txt` | 运行时依赖 + Nuitka 打包依赖（仍是 CPU） |
| `models/vda_vits_t1.onnx` | INT8 T=1 vits |
| `models/export_meta.txt` | 导出元数据 |
| `models/inferno.npy` | inferno LUT 回退 |
| `scripts/build_windows_portable.ps1` | 组装 Windows 便携目录（参考） |
| `scripts/build_nuitka.sh` | Nuitka onedir 打包（参考） |
| `.github/workflows/build-windows.yml` | 便携打包 Actions（参考） |

## 体积说明

日常请用源码安装。若有人问「GPU exe 会不会更大」——会，而且明显更大。下面是已知数字，没有实测过完整 GPU exe：

- 历史上 CPU 版 Nuitka GUI 大约 **76 MB**；近期 Linux Nuitka 产物大约 96–98 MB。CPU exe 大致在 **70–100 MB**。
- `onnxruntime-gpu` 的 wheel 本身大约 **Windows 186–221 MB**、**Linux 191–271 MB**（CUDA 13 / 12.8）。
- 若把 GPU wheel 打进 exe（不另打完整 CUDA toolkit），体积大概会到 **250 MB+**，对比 CPU 的约 70–100 MB。
- 若再捆绑独立 CUDA toolkit，还会再大很多（数百 MB 到 1 GB+）。

源码安装 GPU 只是换成 `onnxruntime-gpu` 这一个 pip 包。

## 致谢

深度模型来自 [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) / Depth Anything 系列。本项目把它收成离线、ONNX Runtime、T=1 逐帧的小应用；视频路径没有使用官方 32 帧时序推理。
