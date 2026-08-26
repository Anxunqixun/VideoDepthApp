#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared ONNX T=1 inference (images + per-frame video). No torch."""
from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np

_NCPU = os.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(_NCPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_NCPU))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_NCPU))
os.environ.setdefault("ORT_NUM_THREADS", str(_NCPU))

NET = 518
MAX_RES = 640
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_NAME = "vda_vits_t1.onnx"

_SESSION = None
_SESS_LOCK = threading.Lock()
_INFERNO = None

# OpenCV LUT names. turbo is the default: more even, less crushed-purple than inferno.
COLORMAPS = (
    "turbo",
    "viridis",
    "plasma",
    "magma",
    "inferno",
    "jet",
    "hot",
    "ocean",
    "cool",
    "gray",
)
COLORMAP_LABELS = {
    "turbo": "青橙 turbo（推荐）",
    "viridis": "绿黄 viridis",
    "plasma": "紫粉 plasma",
    "magma": "暗紫 magma",
    "inferno": "热力 inferno（原来）",
    "jet": "彩虹 jet",
    "hot": "红热 hot",
    "ocean": "海洋 ocean",
    "cool": "冷色 cool",
    "gray": "灰度 gray",
}
DEFAULT_COLORMAP = "turbo"


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _bundle_dir():
    # Nuitka onefile extracts next to __file__
    return os.path.dirname(os.path.abspath(__file__))


def find_model():
    names = [
        os.path.join("models", MODEL_NAME),
        MODEL_NAME,
    ]
    roots = [_bundle_dir(), _app_dir(), os.getcwd()]
    seen = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        for rel in names:
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError("找不到 ONNX 模型 %s" % MODEL_NAME)


def _load_inferno():
    global _INFERNO
    if _INFERNO is not None:
        return _INFERNO
    for root in (_bundle_dir(), _app_dir()):
        p = os.path.join(root, "models", "inferno.npy")
        if os.path.isfile(p):
            _INFERNO = np.load(p)
            return _INFERNO
    # fallback grayscale
    g = np.arange(256, dtype=np.uint8)
    _INFERNO = np.stack([g, g, g], axis=1)
    return _INFERNO


def _cv_colormap(name):
    table = {
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "ocean": cv2.COLORMAP_OCEAN,
        "cool": cv2.COLORMAP_COOL,
    }
    return table.get((name or DEFAULT_COLORMAP).lower())


def depth_to_color(depth_hw, colormap=DEFAULT_COLORMAP, invert=False, dmin=None, dmax=None):
    d = np.asarray(depth_hw, dtype=np.float32)
    if dmin is None:
        dmin = float(d.min())
    if dmax is None:
        dmax = float(d.max())
    denom = (dmax - dmin) if dmax > dmin else 1.0
    idx = ((d - dmin) / denom * 255.0).astype(np.uint8)
    if invert:
        idx = 255 - idx
    name = (colormap or DEFAULT_COLORMAP).lower()
    if name in ("gray", "grey", "grayscale"):
        return np.stack([idx, idx, idx], axis=-1)
    cmap = _cv_colormap(name)
    if cmap is None:
        # fall back to bundled inferno LUT
        lut = _load_inferno()
        return lut[idx]
    bgr = cv2.applyColorMap(idx, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def depth_to_inferno(depth_hw):
    return depth_to_color(depth_hw, colormap="inferno")


def save_depth_png(depth_hw, output_path, colormap=DEFAULT_COLORMAP, invert=False):
    vis = depth_to_color(depth_hw, colormap=colormap, invert=invert)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("无法编码 PNG")
    buf.tofile(output_path)
    return vis


def apply_max_res(frame, max_res=MAX_RES):
    h, w = frame.shape[:2]
    if max_res is None or max_res <= 0 or max(h, w) <= max_res:
        return frame
    scale = max_res / float(max(h, w))
    nh = max(int(round(h * scale)) // 2 * 2, 2)
    nw = max(int(round(w * scale)) // 2 * 2, 2)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def letterbox_nchw(rgb_u8, net=NET):
    h, w = rgb_u8.shape[:2]
    scale = net / float(max(h, w))
    nh = max(int(round(h * scale)), 1)
    nw = max(int(round(w * scale)), 1)
    img = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((net, net, 3), dtype=np.uint8)
    top = (net - nh) // 2
    left = (net - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    x = canvas.astype(np.float32) * (1.0 / 255.0)
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x, dtype=np.float32), (top, left, nh, nw, h, w)


def load_session():
    global _SESSION
    with _SESS_LOCK:
        if _SESSION is not None:
            return _SESSION
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = _NCPU
        so.inter_op_num_threads = min(2, _NCPU)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = find_model()
        _SESSION = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        return _SESSION


def infer_prepared(rgb_u8):
    """rgb uint8 at working resolution -> HxW float32 depth."""
    x, box = letterbox_nchw(rgb_u8)
    sess = load_session()
    raw = sess.run(None, {"pixel_values": x})[0][0]
    top, left, nh, nw, oh, ow = box
    crop = raw[top:top + nh, left:left + nw]
    depth = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return np.maximum(depth, 0).astype(np.float32)


def load_rgb_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("无法读取图片: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def infer_one_image(rgb, progress=None, max_res=MAX_RES):
    if progress:
        progress("正在准备图片…")
    rgb = apply_max_res(rgb, max_res)
    if progress:
        progress("正在推理（单帧 T=1 / ONNX）…")
    t0 = time.time()
    load_session()
    depth = infer_prepared(rgb)
    dt = time.time() - t0
    if progress:
        progress("推理完成，用时 %.2f 秒" % dt)
    return depth, rgb, dt


def read_video_frames(video_path, process_length=-1, target_fps=-1, max_res=MAX_RES, frame_stride=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频: %s" % video_path)
    original_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / float(max(original_height, original_width))
        height = max(int(round(original_height * scale)) // 2 * 2, 2)
        width = max(int(round(original_width * scale)) // 2 * 2, 2)
    else:
        height, width = original_height, original_width
    frame_stride = max(int(frame_stride or 1), 1)
    if target_fps is not None and float(target_fps) > 0:
        fps_stride = max(int(round(original_fps / float(target_fps))), 1)
    else:
        fps_stride = 1
    stride = max(fps_stride * frame_stride, 1)
    fps = original_fps / float(stride)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if process_length > 0 and len(frames) >= process_length:
            break
        if frame_count % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[0], frame.shape[1]) != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frame_count += 1
    cap.release()
    if not frames:
        raise RuntimeError("视频没有可读帧")
    return np.stack(frames, axis=0), fps


def infer_one_video(path, frame_stride=2, target_fps=None, progress=None, max_res=MAX_RES):
    if progress:
        progress("正在读取视频…")
    tfps = -1 if not target_fps else float(target_fps)
    frames, out_fps = read_video_frames(
        path, process_length=-1, target_fps=tfps, max_res=max_res, frame_stride=frame_stride
    )
    n = int(frames.shape[0])
    if progress:
        progress("已读取 %d 帧，输出约 %.2f fps，开始逐帧推理…" % (n, out_fps))
    load_session()
    t0 = time.time()
    depths = []
    for i in range(n):
        if progress and (i % 5 == 0 or i + 1 == n):
            progress("推理帧 %d / %d（逐帧 ONNX，无时序窗）" % (i + 1, n))
        depths.append(infer_prepared(frames[i]))
    dt = time.time() - t0
    if progress:
        progress("视频推理完成，用时 %.1f 秒" % dt)
    return np.stack(depths, axis=0), out_fps, frames, dt


def save_video(frames, output_video_path, fps=10, is_depths=False, colormap=DEFAULT_COLORMAP, invert=False):
    if is_depths:
        dmin, dmax = float(frames.min()), float(frames.max())
        vis_list = [
            depth_to_color(frames[i], colormap=colormap, invert=invert, dmin=dmin, dmax=dmax)
            for i in range(frames.shape[0])
        ]
        frames_out = vis_list
    else:
        frames_out = [frames[i] for i in range(frames.shape[0])]
    h, w = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("无法写入视频: %s" % output_video_path)
    for fr in frames_out:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def default_out_name(src_path, kind):
    base = os.path.splitext(os.path.basename(src_path))[0]
    if kind == "image":
        return base + "_depth.png"
    return base + "_depth.mp4"
