# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import cv2
import matplotlib.cm as cm
import imageio
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1, frame_stride=1):
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        orig_fps = float(vid.get_avg_fps()) or 30.0
        frame_stride = max(int(frame_stride or 1), 1)
        if target_fps is not None and float(target_fps) > 0:
            fps_stride = max(int(round(orig_fps / float(target_fps))), 1)
        else:
            fps_stride = 1
        stride = max(fps_stride * frame_stride, 1)
        fps = orig_fps / float(stride)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        frames = vid.get_batch(frames_idx).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        orig_fps = float(original_fps) or 30.0
        frame_stride = max(int(frame_stride or 1), 1)
        if target_fps is not None and float(target_fps) > 0:
            fps_stride = max(int(round(orig_fps / float(target_fps))), 1)
        else:
            fps_stride = 1
        stride = max(fps_stride * frame_stride, 1)
        fps = orig_fps / float(stride)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (process_length > 0 and frame_count >= process_length):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        import matplotlib
        try:
            cmap_obj = matplotlib.colormaps['inferno']
        except Exception:
            cmap_obj = matplotlib.cm.get_cmap("inferno")
        if hasattr(cmap_obj, 'colors') and cmap_obj.colors is not None:
            colormap = np.array(cmap_obj.colors)
        else:
            colormap = np.array(cmap_obj(np.linspace(0, 1, 256)))[:, :3]
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])

    writer.close()

def apply_max_res(frame, max_res=640):
    """Downscale so max(h, w) <= max_res. Never upsamples."""
    h, w = frame.shape[:2]
    if max_res is None or max_res <= 0 or max(h, w) <= max_res:
        return frame
    scale = max_res / float(max(h, w))
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    nh = max(nh - (nh % 2), 2)
    nw = max(nw - (nw % 2), 2)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def depth_to_inferno(depth_hw):
    """HxW float depth -> HxWx3 uint8 inferno visualization (same as save_video is_depths)."""
    import matplotlib
    try:
        cmap_obj = matplotlib.colormaps['inferno']
    except Exception:
        cmap_obj = matplotlib.cm.get_cmap("inferno")
    if hasattr(cmap_obj, 'colors') and cmap_obj.colors is not None:
        colormap = np.array(cmap_obj.colors)
    else:
        colormap = np.array(cmap_obj(np.linspace(0, 1, 256)))[:, :3]
    d_min, d_max = float(depth_hw.min()), float(depth_hw.max())
    denom = (d_max - d_min) if d_max > d_min else 1.0
    depth_norm = ((depth_hw - d_min) / denom * 255.0).astype(np.uint8)
    return (colormap[depth_norm] * 255).astype(np.uint8)


def save_depth_png(depth_hw, output_path):
    vis = depth_to_inferno(depth_hw)
    imageio.imwrite(output_path, vis)
    return vis

