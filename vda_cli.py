#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vda-cli: ONNX CPU depth, no GUI."""
from __future__ import annotations

import argparse
import os
import sys

# Match engine default (30% of cores) before importing cv2/numpy via engine.
_NCPU = os.cpu_count() or 8
_DEFAULT_THREADS = max(1, int(round(_NCPU * 30 / 100.0)))
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))

from engine import (  # noqa: E402
    COLORMAPS,
    DEFAULT_COLORMAP,
    DEFAULT_CPU_PERCENT,
    clamp_cpu_percent,
    default_out_name,
    infer_one_image,
    infer_one_video,
    load_rgb_image,
    load_session,
    save_depth_png,
    save_video,
    threads_from_percent,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="深度估计 CLI (ONNX CPU)")
    p.add_argument("--image", help="处理单张图片并退出")
    p.add_argument("--video", help="处理视频并退出")
    p.add_argument("--output", "-o", help="输出文件路径")
    p.add_argument("--stride", "--frame-stride", type=int, default=2, dest="stride")
    p.add_argument("--fps", "--target-fps", type=float, default=None, dest="fps")
    p.add_argument("--colormap", "-c", default=DEFAULT_COLORMAP, choices=list(COLORMAPS),
                   help="深度配色，默认 turbo")
    p.add_argument("--invert", action="store_true", help="反转近远颜色")
    p.add_argument(
        "--cpu-percent", type=float, default=DEFAULT_CPU_PERCENT, metavar="N",
        help="CPU 占用百分比 1–100（默认 %d，折算线程数至少为 1）" % DEFAULT_CPU_PERCENT,
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.image and not args.video:
        parse_args(["-h"])
        return 2
    cpu_percent = clamp_cpu_percent(args.cpu_percent)
    n_threads = threads_from_percent(cpu_percent)
    print("CPU 占用 %d%% → %d 线程 / %d 核" % (cpu_percent, n_threads, _NCPU), flush=True)
    load_session(cpu_percent=cpu_percent)
    if args.image:
        src = os.path.abspath(args.image)
        rgb = load_rgb_image(src)
        out = args.output or os.path.join(os.path.dirname(src), default_out_name(src, "image"))
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        depth, _, dt = infer_one_image(
            rgb, progress=lambda s: print(s, flush=True), cpu_percent=cpu_percent,
        )
        save_depth_png(depth, out, colormap=args.colormap, invert=args.invert)
        print("WROTE", out, "time_s", round(dt, 3), flush=True)
        return 0
    src = os.path.abspath(args.video)
    out = args.output or os.path.join(os.path.dirname(src), default_out_name(src, "video"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    depths, fps, _, dt = infer_one_video(
        src, frame_stride=args.stride, target_fps=args.fps,
        progress=lambda s: print(s, flush=True),
        cpu_percent=cpu_percent,
    )
    save_video(depths, out, fps=fps, is_depths=True, colormap=args.colormap, invert=args.invert)
    print("WROTE", out, "time_s", round(dt, 3), "fps", fps, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
