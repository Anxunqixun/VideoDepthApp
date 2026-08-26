#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video-Depth-Anything desktop app (CPU, vits, offline)."""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback

# Threads before importing torch
_NCPU = os.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(_NCPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_NCPU))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_NCPU))


def _frozen():
    return bool(getattr(sys, "frozen", False))


def _meipass():
    return getattr(sys, "_MEIPASS", None)


def _app_dir():
    if _frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _vendor_candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    out = [
        os.path.join(_app_dir(), "vendor"),
        os.path.join(here, "vendor"),
    ]
    mp = _meipass()
    if mp:
        out.extend([os.path.join(mp, "vendor"), mp])
    # de-dup
    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def _setup_tcl():
    """Point Tk at bundled Tcl/Tk 9 when frozen (uv CPython is built against 9.0)."""
    roots = []
    mp = _meipass()
    if mp:
        roots.append(mp)
    roots.append(_app_dir())
    roots.append(os.path.join(_app_dir(), "_internal"))
    for root in roots:
        tcl = os.path.join(root, "tcl9.0")
        tk = os.path.join(root, "tk9.0")
        if os.path.isdir(tcl):
            os.environ.setdefault("TCL_LIBRARY", tcl)
        if os.path.isdir(tk):
            os.environ.setdefault("TK_LIBRARY", tk)
        if os.path.isdir(tcl) and os.path.isdir(tk):
            break


def _setup_paths():
    for p in _vendor_candidates():
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return _vendor_candidates()


_setup_tcl()
_setup_paths()

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
import torch

torch.set_num_threads(_NCPU)
try:
    torch.set_num_interop_threads(min(2, _NCPU))
except Exception:
    pass

from utils.dc_utils import (  # noqa: E402
    apply_max_res,
    depth_to_inferno,
    read_video_frames,
    save_depth_png,
    save_video,
)
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

MODEL_CFG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
CKPT_NAME = "video_depth_anything_vits.pth"
INPUT_SIZE = 518
MAX_RES = 640
DEVICE = "cpu"
FP32 = True


def find_checkpoint():
    names = [
        os.path.join("checkpoints", CKPT_NAME),
        os.path.join("vendor", "checkpoints", CKPT_NAME),
        CKPT_NAME,
    ]
    roots = _vendor_candidates() + [_app_dir(), _meipass() or ""]
    for root in roots:
        if not root:
            continue
        for rel in names:
            path = os.path.join(root, rel)
            if os.path.isfile(path):
                return path
            # also if root already is vendor/
            path2 = os.path.join(root, "checkpoints", CKPT_NAME)
            if os.path.isfile(path2):
                return path2
    raise FileNotFoundError(
        "找不到权重文件 %s。已搜索: %s" % (CKPT_NAME, ", ".join(roots))
    )


_MODEL = None
_MODEL_LOCK = threading.Lock()


def load_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        ckpt = find_checkpoint()
        model = VideoDepthAnything(**MODEL_CFG)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model = model.to(DEVICE).eval()
        _MODEL = model
        return _MODEL


def load_rgb_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("无法读取图片: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def infer_one_image(rgb, progress=None):
    if progress:
        progress("正在准备图片…")
    rgb = apply_max_res(rgb, MAX_RES)
    if progress:
        progress("正在推理（单帧 T=1）…")
    t0 = time.time()
    model = load_model()
    depth = model.infer_image(rgb, input_size=INPUT_SIZE, device=DEVICE, fp32=FP32)
    dt = time.time() - t0
    if progress:
        progress("推理完成，用时 %.2f 秒" % dt)
    return depth, rgb, dt


def infer_one_video(path, frame_stride=2, target_fps=None, progress=None):
    if progress:
        progress("正在读取视频…")
    tfps = -1 if not target_fps else float(target_fps)
    frames, out_fps = read_video_frames(
        path, process_length=-1, target_fps=tfps, max_res=MAX_RES, frame_stride=frame_stride
    )
    if frames is None or len(frames) == 0:
        raise RuntimeError("视频没有可读帧")
    n = int(frames.shape[0])
    if progress:
        progress("已读取 %d 帧，输出约 %.2f fps，开始推理…" % (n, out_fps))

    def cb(i, total):
        if progress:
            progress("推理窗口 %d / %d（每窗 32 帧）" % (i, total))

    model = load_model()
    t0 = time.time()
    depths, fps = model.infer_video_depth(
        frames,
        out_fps,
        input_size=INPUT_SIZE,
        device=DEVICE,
        fp32=FP32,
        progress_callback=cb,
    )
    dt = time.time() - t0
    if progress:
        progress("视频推理完成，用时 %.1f 秒" % dt)
    return depths, fps, frames, dt


def default_out_name(src_path, kind):
    base = os.path.splitext(os.path.basename(src_path))[0]
    if kind == "image":
        return base + "_depth.png"
    return base + "_depth.mp4"


def run_cli(args):
    load_model()
    if args.image:
        src = os.path.abspath(args.image)
        rgb = load_rgb_image(src)
        out = args.output or os.path.join(os.path.dirname(src), default_out_name(src, "image"))
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        depth, _, dt = infer_one_image(rgb, progress=lambda s: print(s, flush=True))
        save_depth_png(depth, out)
        print("WROTE", out, "time_s", round(dt, 3), flush=True)
        return 0
    if args.video:
        src = os.path.abspath(args.video)
        out = args.output or os.path.join(os.path.dirname(src), default_out_name(src, "video"))
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        depths, fps, _, dt = infer_one_video(
            src,
            frame_stride=args.frame_stride,
            target_fps=args.target_fps,
            progress=lambda s: print(s, flush=True),
        )
        save_video(depths, out, fps=fps, is_depths=True)
        print("WROTE", out, "time_s", round(dt, 3), "fps", fps, flush=True)
        return 0
    print("需要 --image 或 --video，或直接启动图形界面", file=sys.stderr)
    return 2


def _pick_cjk_font(root):
    import tkinter.font as tkfont

    families = set(tkfont.families(root))
    for name in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
    ):
        if name in families:
            return name
    return None


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("深度估计")
            self.geometry("780x720")
            self.minsize(640, 560)
            self.msg_q = queue.Queue()
            self.worker = None
            self.input_path = ""
            self.input_kind = None  # 'image' | 'video'
            self.preview_photo = None
            self._build()
            self.after(120, self._poll)
            self.after(200, self._preload_model)

        def _build(self):
            font_name = _pick_cjk_font(self)
            ui_font = (font_name, 11) if font_name else None
            if ui_font:
                self.option_add("*Font", ui_font)

            pad = {"padx": 10, "pady": 4}
            frm = ttk.Frame(self)
            frm.pack(fill=tk.X, **pad)

            ttk.Label(frm, text="输入文件").grid(row=0, column=0, sticky="w")
            self.var_input = tk.StringVar()
            ttk.Entry(frm, textvariable=self.var_input, width=56).grid(
                row=0, column=1, columnspan=2, sticky="ew", padx=4
            )
            ttk.Button(frm, text="选择图片", command=self._pick_image).grid(row=0, column=3, padx=2)
            ttk.Button(frm, text="选择视频", command=self._pick_video).grid(row=0, column=4, padx=2)

            ttk.Label(frm, text="输出文件夹").grid(row=1, column=0, sticky="w")
            self.var_outdir = tk.StringVar()
            ttk.Entry(frm, textvariable=self.var_outdir, width=56).grid(
                row=1, column=1, columnspan=2, sticky="ew", padx=4
            )
            ttk.Button(frm, text="浏览", command=self._pick_outdir).grid(row=1, column=3, padx=2)

            ttk.Label(frm, text="抽帧间隔 N").grid(row=2, column=0, sticky="w")
            self.var_stride = tk.StringVar(value="2")
            ttk.Entry(frm, textvariable=self.var_stride, width=8).grid(row=2, column=1, sticky="w", padx=4)
            ttk.Label(frm, text="每 N 帧取 1 帧（默认 2，仅视频）").grid(row=2, column=2, columnspan=2, sticky="w")

            ttk.Label(frm, text="目标帧率").grid(row=3, column=0, sticky="w")
            self.var_fps = tk.StringVar(value="")
            ttk.Entry(frm, textvariable=self.var_fps, width=8).grid(row=3, column=1, sticky="w", padx=4)
            ttk.Label(frm, text="可空=不限；填写则再按目标帧率抽帧").grid(row=3, column=2, columnspan=2, sticky="w")

            hint = (
                "抽帧规则：先按间隔 N 取样；若同时填写目标帧率，则综合间隔 = N × round(原fps/目标fps)。"
                " 输出帧率 = 原fps / 综合间隔。图片走单帧 T=1，不填充 32 帧。"
            )
            ttk.Label(frm, text=hint, wraplength=740, justify="left").grid(
                row=4, column=0, columnspan=5, sticky="w", pady=(2, 6)
            )

            btnrow = ttk.Frame(frm)
            btnrow.grid(row=5, column=0, columnspan=5, sticky="w", pady=6)
            self.btn_start = ttk.Button(btnrow, text="开始", command=self._start)
            self.btn_start.pack(side=tk.LEFT)
            self.var_status = tk.StringVar(value="就绪（首次推理会加载模型）")
            ttk.Label(btnrow, textvariable=self.var_status, wraplength=600).pack(side=tk.LEFT, padx=12)

            frm.columnconfigure(1, weight=1)

            prev = ttk.LabelFrame(self, text="深度预览")
            prev.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
            self.preview = tk.Label(prev, text="完成后在此显示深度图；视频会提示输出 mp4 路径", anchor="center")
            self.preview.pack(fill=tk.BOTH, expand=True)

        def _preload_model(self):
            def work():
                try:
                    self.msg_q.put(("progress", "正在加载模型…"))
                    load_model()
                    self.msg_q.put(("progress", "模型已就绪，请选择图片或视频"))
                except Exception as e:
                    self.msg_q.put(("error", "模型加载失败: %s" % e))

            threading.Thread(target=work, daemon=True).start()

        def _pick_image(self):
            path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[
                    ("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                    ("全部", "*.*"),
                ],
            )
            if not path:
                return
            self.input_path = path
            self.input_kind = "image"
            self.var_input.set(path)
            if not self.var_outdir.get().strip():
                self.var_outdir.set(os.path.dirname(path))

        def _pick_video(self):
            path = filedialog.askopenfilename(
                title="选择视频",
                filetypes=[
                    ("视频", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                    ("全部", "*.*"),
                ],
            )
            if not path:
                return
            self.input_path = path
            self.input_kind = "video"
            self.var_input.set(path)
            if not self.var_outdir.get().strip():
                self.var_outdir.set(os.path.dirname(path))

        def _pick_outdir(self):
            path = filedialog.askdirectory(title="选择输出文件夹")
            if path:
                self.var_outdir.set(path)

        def _parse_stride(self):
            raw = self.var_stride.get().strip() or "2"
            n = int(float(raw))
            if n < 1:
                raise ValueError("抽帧间隔必须 >= 1")
            return n

        def _parse_fps(self):
            raw = self.var_fps.get().strip()
            if raw == "":
                return None
            v = float(raw)
            if v <= 0:
                raise ValueError("目标帧率必须为正数，或留空")
            return v

        def _start(self):
            if self.worker and self.worker.is_alive():
                messagebox.showinfo("提示", "正在处理，请稍候")
                return
            path = self.var_input.get().strip() or self.input_path
            if not path:
                messagebox.showwarning("提示", "请先选择图片或视频")
                return
            if not os.path.isfile(path):
                messagebox.showerror("错误", "文件不存在: %s" % path)
                return
            kind = self.input_kind
            if kind is None:
                ext = os.path.splitext(path)[1].lower()
                kind = "image" if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"} else "video"
            outdir = self.var_outdir.get().strip() or os.path.dirname(path)
            os.makedirs(outdir, exist_ok=True)
            try:
                stride = self._parse_stride()
                tfps = self._parse_fps()
            except Exception as e:
                messagebox.showerror("参数错误", str(e))
                return

            self.btn_start.configure(state="disabled")
            self.var_status.set("启动中…")

            def work():
                try:
                    load_model()
                    if kind == "image":
                        rgb = load_rgb_image(path)
                        depth, _, dt = infer_one_image(rgb, progress=lambda s: self.msg_q.put(("progress", s)))
                        out = os.path.join(outdir, default_out_name(path, "image"))
                        vis = save_depth_png(depth, out)
                        self.msg_q.put(("done_image", out, vis, dt))
                    else:
                        depths, fps, _, dt = infer_one_video(
                            path,
                            frame_stride=stride,
                            target_fps=tfps,
                            progress=lambda s: self.msg_q.put(("progress", s)),
                        )
                        out = os.path.join(outdir, default_out_name(path, "video"))
                        self.msg_q.put(("progress", "正在保存深度视频…"))
                        save_video(depths, out, fps=fps, is_depths=True)
                        preview = depth_to_inferno(depths[len(depths) // 2])
                        self.msg_q.put(("done_video", out, preview, dt, int(depths.shape[0]), fps))
                except Exception:
                    self.msg_q.put(("error", traceback.format_exc()))

            self.worker = threading.Thread(target=work, daemon=True)
            self.worker.start()

        def _show_preview(self, vis_rgb):
            h, w = vis_rgb.shape[:2]
            box_w = max(self.preview.winfo_width(), 400)
            box_h = max(self.preview.winfo_height(), 300)
            scale = min(box_w / float(w), box_h / float(h), 1.0)
            nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
            vis = cv2.resize(vis_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            ppm = ("P6\n%d %d\n255\n" % (nw, nh)).encode("ascii") + vis.tobytes()
            photo = tk.PhotoImage(data=ppm)
            self.preview_photo = photo
            self.preview.configure(image=photo, text="")

        def _poll(self):
            try:
                while True:
                    msg = self.msg_q.get_nowait()
                    kind = msg[0]
                    if kind == "progress":
                        self.var_status.set(msg[1])
                    elif kind == "done_image":
                        out, vis, dt = msg[1], msg[2], msg[3]
                        self.var_status.set("完成：%s（%.2f 秒）" % (out, dt))
                        self._show_preview(vis)
                        self.btn_start.configure(state="normal")
                    elif kind == "done_video":
                        out, preview, dt, n, fps = msg[1], msg[2], msg[3], msg[4], msg[5]
                        self.var_status.set("完成：%s（%d 帧，%.2f fps，%.1f 秒）" % (out, n, fps, dt))
                        self._show_preview(preview)
                        self.btn_start.configure(state="normal")
                    elif kind == "error":
                        self.var_status.set("出错")
                        self.btn_start.configure(state="normal")
                        messagebox.showerror("出错", msg[1][-2000:])
            except queue.Empty:
                pass
            self.after(100, self._poll)

    app = App()
    app.mainloop()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="深度估计")
    p.add_argument("--image", help="处理单张图片并退出")
    p.add_argument("--video", help="处理视频并退出")
    p.add_argument("--output", "-o", help="输出文件路径")
    p.add_argument("--frame-stride", type=int, default=2, dest="frame_stride")
    p.add_argument("--target-fps", type=float, default=None, dest="target_fps")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.image or args.video:
        return run_cli(args)
    run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
