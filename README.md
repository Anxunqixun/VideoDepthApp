# 深度估计 VideoDepthApp

离线桌面程序：选图片出深度图，选视频可设抽帧间隔 / 目标帧率。CPU，无需联网。

Windows exe 由 GitHub Actions 自动打包。打开仓库 **Actions → Build Windows EXE**，完成后在该次运行的 **Artifacts** 里下载 `VideoDepthApp-windows`。

源码运行（需要先自己装 CPU 版 PyTorch）：

```bash
pip install -r requirements-cpu.txt
# 把 video_depth_anything_vits.pth 放到 vendor/checkpoints/
python app.py
```
