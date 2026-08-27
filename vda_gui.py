#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vda-gui: Chinese tkinter UI, same engine as vda-cli."""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback

# Match engine default (30% of cores) before importing cv2/numpy via engine.
_NCPU = os.cpu_count() or 8
_DEFAULT_THREADS = max(1, int(round(_NCPU * 30 / 100.0)))
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))


def _frozen():
    return bool(getattr(sys, "frozen", False))


def _app_dir():
    if _frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _setup_tcl():
    roots = [os.path.dirname(os.path.abspath(__file__)), _app_dir()]
    for root in roots:
        for tcln, tkn in (("tcl9.0", "tk9.0"), ("tcl8.6", "tk8.6")):
            tcl = os.path.join(root, tcln)
            tk = os.path.join(root, tkn)
            if os.path.isdir(tcl):
                os.environ.setdefault("TCL_LIBRARY", tcl)
            if os.path.isdir(tk):
                os.environ.setdefault("TK_LIBRARY", tk)
            if os.path.isdir(tcl) and os.path.isdir(tk):
                return


_setup_tcl()

from engine import (  # noqa: E402
    COLORMAP_LABELS,
    COLORMAPS,
    DEFAULT_COLORMAP,
    DEFAULT_CPU_PERCENT,
    clamp_cpu_percent,
    cpu_percent_label,
    default_out_name,
    depth_to_color,
    infer_one_image,
    infer_one_video,
    load_rgb_image,
    load_session,
    save_depth_png,
    save_video,
)


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
    import cv2

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("深度估计")
            self.geometry("780x760")
            self.minsize(640, 600)
            self.msg_q = queue.Queue()
            self.worker = None
            self.input_path = ""
            self.input_kind = None
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

            ttk.Label(frm, text="深度配色").grid(row=4, column=0, sticky="w")
            self.var_cmap = tk.StringVar(value=COLORMAP_LABELS[DEFAULT_COLORMAP])
            self.cmb_cmap = ttk.Combobox(
                frm, textvariable=self.var_cmap, width=22, state="readonly",
                values=[COLORMAP_LABELS[k] for k in COLORMAPS],
            )
            self.cmb_cmap.grid(row=4, column=1, sticky="w", padx=4)
            self.var_invert = tk.BooleanVar(value=False)
            ttk.Checkbutton(frm, text="反转近远（近处变暗）", variable=self.var_invert).grid(
                row=4, column=2, columnspan=2, sticky="w"
            )

            ttk.Label(frm, text="CPU 占用").grid(row=5, column=0, sticky="w")
            cpu_row = ttk.Frame(frm)
            cpu_row.grid(row=5, column=1, columnspan=4, sticky="ew", padx=4)
            self._cpu_updating = False
            self.var_cpu = tk.IntVar(value=DEFAULT_CPU_PERCENT)
            self.var_cpu_hint = tk.StringVar()
            self.spn_cpu = ttk.Spinbox(
                cpu_row, from_=1, to=100, textvariable=self.var_cpu, width=5,
                command=self._on_cpu_change,
            )
            self.spn_cpu.pack(side=tk.LEFT)
            ttk.Label(cpu_row, text="%").pack(side=tk.LEFT, padx=(2, 8))
            self.sld_cpu = ttk.Scale(
                cpu_row, from_=1, to=100, variable=self.var_cpu,
                command=self._on_cpu_scale, length=180,
            )
            self.sld_cpu.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(cpu_row, textvariable=self.var_cpu_hint).pack(side=tk.LEFT, padx=(10, 0))
            self.var_cpu.trace_add("write", self._on_cpu_change)
            self._on_cpu_change()

            hint = (
                "抽帧规则：先按间隔 N 取样；若同时填写目标帧率，则综合间隔 = N × round(原fps/目标fps)。"
                " 输出帧率 = 原fps / 综合间隔。图片走单帧 T=1。视频为逐帧 ONNX（优先体积与速度，无 32 帧时序窗）。"
                " CPU 占用限制推理线程数，默认 30%，不会占满全部核心。"
            )
            ttk.Label(frm, text=hint, wraplength=740, justify="left").grid(
                row=6, column=0, columnspan=5, sticky="w", pady=(2, 6)
            )

            btnrow = ttk.Frame(frm)
            btnrow.grid(row=7, column=0, columnspan=5, sticky="w", pady=6)
            self.btn_start = ttk.Button(btnrow, text="开始", command=self._start)
            self.btn_start.pack(side=tk.LEFT)
            self.var_status = tk.StringVar(value="就绪（首次推理会加载模型）")
            ttk.Label(btnrow, textvariable=self.var_status, wraplength=600).pack(side=tk.LEFT, padx=12)
            frm.columnconfigure(1, weight=1)

            prev = ttk.LabelFrame(self, text="深度预览")
            prev.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
            self.preview = tk.Label(prev, text="完成后在此显示深度图；视频会提示输出 mp4 路径", anchor="center")
            self.preview.pack(fill=tk.BOTH, expand=True)

        def _set_cpu_percent(self, raw):
            p = clamp_cpu_percent(raw)
            if self._cpu_updating:
                self.var_cpu_hint.set(cpu_percent_label(p))
                return p
            self._cpu_updating = True
            try:
                try:
                    cur = int(round(float(self.var_cpu.get())))
                except Exception:
                    cur = None
                if cur != p:
                    self.var_cpu.set(p)
                self.var_cpu_hint.set(cpu_percent_label(p))
            finally:
                self._cpu_updating = False
            return p

        def _on_cpu_scale(self, value=None):
            self._set_cpu_percent(value if value is not None else self.var_cpu.get())

        def _on_cpu_change(self, *_args):
            if self._cpu_updating:
                return
            try:
                self._set_cpu_percent(self.var_cpu.get())
            except Exception:
                self._set_cpu_percent(DEFAULT_CPU_PERCENT)

        def _parse_cpu(self):
            return self._set_cpu_percent(self.var_cpu.get())

        def _preload_model(self):
            def work():
                try:
                    self.msg_q.put(("progress", "正在加载模型…"))
                    load_session(cpu_percent=DEFAULT_CPU_PERCENT)
                    self.msg_q.put(("progress", "模型已就绪，请选择图片或视频"))
                except Exception as e:
                    self.msg_q.put(("error", "模型加载失败: %s" % e))
            threading.Thread(target=work, daemon=True).start()

        def _pick_image(self):
            path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("全部", "*.*")],
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
                filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("全部", "*.*")],
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

        def _parse_cmap(self):
            label = self.var_cmap.get().strip()
            for key, lab in COLORMAP_LABELS.items():
                if lab == label or key == label.lower():
                    return key
            return DEFAULT_COLORMAP

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
                cmap = self._parse_cmap()
                invert = bool(self.var_invert.get())
                cpu_percent = self._parse_cpu()
            except Exception as e:
                messagebox.showerror("参数错误", str(e))
                return
            self.btn_start.configure(state="disabled")
            self.var_status.set("启动中…")

            def work():
                try:
                    load_session(cpu_percent=cpu_percent)
                    if kind == "image":
                        rgb = load_rgb_image(path)
                        depth, _, dt = infer_one_image(
                            rgb, progress=lambda s: self.msg_q.put(("progress", s)),
                            cpu_percent=cpu_percent,
                        )
                        out = os.path.join(outdir, default_out_name(path, "image"))
                        vis = save_depth_png(depth, out, colormap=cmap, invert=invert)
                        self.msg_q.put(("done_image", out, vis, dt))
                    else:
                        depths, fps, _, dt = infer_one_video(
                            path, frame_stride=stride, target_fps=tfps,
                            progress=lambda s: self.msg_q.put(("progress", s)),
                            cpu_percent=cpu_percent,
                        )
                        out = os.path.join(outdir, default_out_name(path, "video"))
                        self.msg_q.put(("progress", "正在保存深度视频…"))
                        save_video(depths, out, fps=fps, is_depths=True, colormap=cmap, invert=invert)
                        preview = depth_to_color(depths[len(depths) // 2], colormap=cmap, invert=invert)
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


def main():
    run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
