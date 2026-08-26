#!/usr/bin/env python3
from __future__ import annotations
import math, os, sys, traceback, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
VDA = "/workspace/Video-Depth-Anything"
sys.path.insert(0, VDA)
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.dinov2 import DinoVisionTransformer

MODEL_CFG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
CKPT = os.path.join(VDA, "checkpoints", "video_depth_anything_vits.pth")
OUT_DIR = os.path.join(ROOT, "models")
NET = 518


def patch_interpolate_pos_encoding():
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        n_patches = patch_pos_embed.shape[1]
        sqrt_n = int(math.sqrt(int(n_patches)))
        if int(w0) == sqrt_n and int(h0) == sqrt_n:
            return self.pos_embed.to(previous_dtype)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_n, sqrt_n, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(w0, h0), mode="bicubic", align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    DinoVisionTransformer.interpolate_pos_encoding = interpolate_pos_encoding


class ImageDepthT1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.unsqueeze(1)).squeeze(1)


def load_model():
    patch_interpolate_pos_encoding()
    model = VideoDepthAnything(**MODEL_CFG)
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def check_ort(onnx_path, dummy_np, torch_out):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pixel_values": dummy_np})[0]
    t = torch_out.detach().cpu().numpy() if torch.is_tensor(torch_out) else torch_out
    mae = float(np.mean(np.abs(out - t)))
    mx = float(np.max(np.abs(out - t)))
    print("ORT", os.path.basename(onnx_path), "mae", round(mae, 6), "max", round(mx, 6),
          "shape", out.shape, "MB", round(os.path.getsize(onnx_path) / 1e6, 2))
    return mae, mx, out


def try_fp16(src, dst):
    import onnx
    from onnxconverter_common import float16
    model = onnx.load(src)
    for kwargs in (
        dict(keep_io_types=True, disable_shape_infer=True),
        dict(keep_io_types=True),
        dict(keep_io_types=False, disable_shape_infer=True),
    ):
        try:
            print("fp16 try", kwargs)
            m = float16.convert_float_to_float16(model, **kwargs)
            onnx.save(m, dst)
            return dst
        except Exception as e:
            print("fp16 variant failed", type(e).__name__, e)
    raise RuntimeError("all fp16 variants failed")


def preprocess_letterbox(rgb_u8, net=NET):
    h, w = rgb_u8.shape[:2]
    scale = net / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh = max(nh - (nh % 2), 2)
    nw = max(nw - (nw % 2), 2)
    import cv2
    img = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((net, net, 3), dtype=np.uint8)
    top = (net - nh) // 2
    left = (net - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    x = canvas.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
    return x, (top, left, nh, nw, h, w)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading torch model...")
    model = load_model()
    wrapper = ImageDepthT1(model).eval()
    dummy = torch.randn(1, 3, NET, NET, dtype=torch.float32)
    with torch.inference_mode():
        torch_out = wrapper(dummy)
    print("torch out", tuple(torch_out.shape), float(torch_out.min()), float(torch_out.max()))
    dummy_np = dummy.numpy()
    fp32_path = os.path.join(OUT_DIR, "vda_vits_t1_fp32.onnx")
    fp16_path = os.path.join(OUT_DIR, "vda_vits_t1_fp16.onnx")
    int8_path = os.path.join(OUT_DIR, "vda_vits_t1_int8.onnx")
    print("exporting fixed %dx%d fp32..." % (NET, NET))
    torch.onnx.export(
        wrapper, dummy, fp32_path,
        input_names=["pixel_values"], output_names=["depth"],
        opset_version=17, do_constant_folding=True,
    )
    print("exported", fp32_path, os.path.getsize(fp32_path))
    mae32, mx32, _ = check_ort(fp32_path, dummy_np, torch_out)
    chosen, kind = fp32_path, "fp32"
    try:
        try_fp16(fp32_path, fp16_path)
        mae16, mx16, _ = check_ort(fp16_path, dummy_np, torch_out)
        if mx16 < 2.0:
            chosen, kind = fp16_path, "fp16"
            print("fp16 accepted")
    except Exception:
        traceback.print_exc()
        print("fp16 failed")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        mae8, mx8, _ = check_ort(int8_path, dummy_np, torch_out)
        print("int8 mae", mae8, "max", mx8)
    except Exception:
        traceback.print_exc()

    # real-image compare vs official infer_image (letterbox vs official keep-aspect)
    import cv2
    cap = cv2.VideoCapture("/workspace/test-input-2.mp4")
    ok, bgr = cap.read()
    cap.release()
    assert ok
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # max_res 640
    h, w = rgb.shape[:2]
    if max(h, w) > 640:
        s = 640 / float(max(h, w))
        rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    print("real rgb", rgb.shape)
    with torch.inference_mode():
        official = model.infer_image(rgb, input_size=518, device="cpu", fp32=True)
    x, box = preprocess_letterbox(rgb)
    import onnxruntime as ort
    for name, path in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
        if not os.path.isfile(path):
            continue
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            raw = sess.run(None, {"pixel_values": x})[0][0]
            top, left, nh, nw, oh, ow = box
            crop = raw[top:top + nh, left:left + nw]
            depth = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
            depth = np.maximum(depth, 0)
            mae = float(np.mean(np.abs(depth - official)))
            rel = mae / (float(official.mean()) + 1e-6)
            print("REAL", name, "vs official mae", round(mae, 4), "rel", round(rel, 4),
                  "MB", round(os.path.getsize(path) / 1e6, 2))
        except Exception as e:
            print("REAL", name, "FAIL", e)

    # Prefer smallest with real-image relative error < 0.15, else fp32
    order = []
    if os.path.isfile(int8_path):
        order.append(("int8", int8_path))
    if os.path.isfile(fp16_path):
        order.append(("fp16", fp16_path))
    order.append(("fp32", fp32_path))
    # Re-eval quickly to pick
    picked = ("fp32", fp32_path)
    sess_cache = {}
    for name, path in order:
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            raw = sess.run(None, {"pixel_values": x})[0][0]
            top, left, nh, nw, oh, ow = box
            crop = raw[top:top + nh, left:left + nw]
            depth = np.maximum(cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR), 0)
            rel = float(np.mean(np.abs(depth - official))) / (float(official.mean()) + 1e-6)
            print("PICK-CHECK", name, "rel", round(rel, 4))
            if rel < 0.15:
                picked = (name, path)
                break
        except Exception as e:
            print("PICK-CHECK", name, e)
    kind, chosen = picked
    final = os.path.join(OUT_DIR, "vda_vits_t1.onnx")
    if os.path.abspath(chosen) != os.path.abspath(final):
        shutil.copy2(chosen, final)
    with open(os.path.join(OUT_DIR, "export_meta.txt"), "w") as f:
        f.write("kind=%s\n" % kind)
        f.write("net=%d\n" % NET)
        f.write("dynamic=False\n")
        f.write("letterbox=1\n")
        f.write("size=%s\n" % os.path.getsize(final))
    print("CHOSEN", kind, final, os.path.getsize(final))


if __name__ == "__main__":
    main()
