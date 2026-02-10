#!/usr/bin/env python3
# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""
Config-style NPY -> Safetensors converter for NSS.

This script keeps the same workflow style as the user-provided version:
  - edit constants in the "配置区"
  - run script directly

Fixes applied over the previous version:
  1) Atomic per-frame loading (avoid list length mismatch / frame misalignment)
  2) Convert motion/jitter from XY to YX channel order
  3) Keep jitter in pixel units (remove "-(jitter - 0.5)" transform)
  4) Use int64 for seq
  5) Compute depth_params from camera defaults instead of a hardcoded placeholder
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from scripts.safetensors_generator.fsr2_methods import depth_to_view_space_params

# ------------------------------------------------------------
# 配置区（只需要改这里）
# ------------------------------------------------------------
DATA_DIR = "/data/vjuicefs_performance_game/public_data/benchmark_q3/250707/ldr_flag_1620x736_noTaa_withMaxLevel_mengde_1/parse_res"
OUTPUT_PATH = "/data/vjuicefs_performance_game/public_data/neural-graphics-model-gym/tests/vnss_mengde1/vnss_sequence_mengde.safetensors"

START_FRAME = None  # None = 自动检测最小帧号
END_FRAME = None  # None = 自动检测最大帧号
MAX_FRAMES = 30  # None = 不限制；例如 30 = 只取前 30 帧

# 数据约定
SCALE = 2.0  # LR -> HR 比例（常见为 2.0）
MOTION_INPUT_ORDER = "xy"  # 你的数据是 xy
JITTER_INPUT_ORDER = "xy"  # 你的数据是 xy

# 如果深度是 reverse-z（近处=1，远处=0）则设 True；否则 False
REVERSE_Z = False

# 可选色彩缩放（默认 1.0，不做放大）
COLOUR_SCALE = 1.0

# 相机参数（用于 depth_params，若无真实元数据，用默认值）
Z_NEAR = 0.1
Z_FAR = 1000.0
FOV_Y_RAD = 1.0471975512  # 60 degrees
INFINITE_Z_FAR = False

# 序列 id（int64）
SEQ_ID = 8897243125409831936

# 文件名模板（与你的数据严格一致）
PATTERNS = {
    "colour": re.compile(r"r_input_color_jittered(\d{5})\.npy"),
    "motion": re.compile(r"r_motion_vectors(\d{5})\.npy"),
    "depth": re.compile(r"r_depth(\d{5})\.npy"),
    "jitter": re.compile(r"r_jitter(\d{5})\.npy"),
}


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def find_frames(data_dir: str) -> List[int]:
    frames = set()
    for fname in os.listdir(data_dir):
        for pat in PATTERNS.values():
            m = pat.match(fname)
            if m:
                frames.add(int(m.group(1)))
    return sorted(frames)


def load_npy_safe(path: str) -> Optional[np.ndarray]:
    """安全加载 .npy 文件，如果缺失返回 None"""
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        return None
    return np.load(path)


def to_chw(array: np.ndarray, channels: int, name: str) -> torch.Tensor:
    """Convert one-frame ndarray to CHW."""
    arr = np.asarray(array)
    if arr.ndim == 3:
        if arr.shape[0] == channels:
            out = arr
        elif arr.shape[-1] == channels:
            out = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(
                f"{name} shape={arr.shape}, expected channel dim={channels} in C-first or C-last."
            )
    elif arr.ndim == 2 and channels == 1:
        out = arr[None, :, :]
    else:
        raise ValueError(f"{name} unsupported shape: {arr.shape}")
    return torch.from_numpy(out).to(torch.float32)


def to_jitter_c11(array: np.ndarray) -> torch.Tensor:
    """Convert one-frame jitter array to (2,1,1)."""
    arr = np.asarray(array)
    if arr.ndim == 1 and arr.shape[0] == 2:
        out = arr[:, None, None]
    elif arr.ndim == 2:
        if arr.shape == (1, 2):
            out = arr.reshape(2, 1, 1)
        elif arr.shape == (2, 1):
            out = arr.reshape(2, 1, 1)
        else:
            raise ValueError(f"jitter unsupported shape: {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape == (2, 1, 1):
            out = arr
        elif arr.shape == (1, 1, 2):
            out = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"jitter unsupported shape: {arr.shape}")
    else:
        raise ValueError(f"jitter unsupported shape: {arr.shape}")
    return torch.from_numpy(out).to(torch.float32)


def xy_to_yx(chw: torch.Tensor, order: str) -> torch.Tensor:
    if order.lower() == "yx":
        return chw
    if order.lower() == "xy":
        return chw[[1, 0], ...]
    raise ValueError(f"Unsupported order: {order}")


def make_depth_params(
    render_size: torch.Tensor,
    z_near: float,
    z_far: float,
    fov_y_rad: float,
    reverse_z: bool,
    infinite_z_far: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = render_size.shape[0]
    device = render_size.device

    z_near_t = torch.full((t, 1), float(z_near), dtype=torch.float32, device=device)
    z_far_t = torch.full((t, 1), float(z_far), dtype=torch.float32, device=device)
    fov_y_t = torch.full((t, 1), float(fov_y_rad), dtype=torch.float32, device=device)
    reverse_z_t = torch.full((t, 1), bool(reverse_z), dtype=torch.bool, device=device)
    infinite_z_far_t = torch.full((t, 1), bool(infinite_z_far), dtype=torch.bool, device=device)

    make_image_like = lambda x: x.unsqueeze(-1).unsqueeze(-1)
    depth_params = depth_to_view_space_params(
        zNear=make_image_like(z_near_t),
        zFar=make_image_like(z_far_t),
        FovY=make_image_like(fov_y_t),
        infinite=make_image_like(infinite_z_far_t),
        renderSizeWidth=make_image_like(render_size[:, 1:2].to(torch.float32)),
        renderSizeHeight=make_image_like(render_size[:, 0:1].to(torch.float32)),
        inverted=make_image_like(reverse_z_t),
    ).squeeze(-1).squeeze(-1)

    return z_near_t, z_far_t, fov_y_t, reverse_z_t, infinite_z_far_t, depth_params


# ------------------------------------------------------------
# 主逻辑
# ------------------------------------------------------------
def main() -> None:
    frames = find_frames(DATA_DIR)
    if not frames:
        raise RuntimeError("❌ No VNSS frames found")

    start = frames[0] if START_FRAME is None else START_FRAME
    end = frames[-1] if END_FRAME is None else END_FRAME
    frame_ids = [f for f in frames if start <= f <= end]

    if MAX_FRAMES is not None:
        frame_ids = frame_ids[:MAX_FRAMES]

    if not frame_ids:
        raise RuntimeError("❌ No frame selected after START/END/MAX filters")

    print(f"✅ Using {len(frame_ids)} frames: {frame_ids[0]} ~ {frame_ids[-1]}")

    colour_list: List[torch.Tensor] = []
    motion_list: List[torch.Tensor] = []
    depth_list: List[torch.Tensor] = []
    jitter_list: List[torch.Tensor] = []

    def p(name: str, fid: int) -> str:
        return os.path.join(DATA_DIR, f"{name}{fid:05d}.npy")

    skipped = 0
    for fid in frame_ids:
        # Atomic loading: this frame is used only if all required files exist
        colour_np = load_npy_safe(p("r_input_color_jittered", fid))
        motion_np = load_npy_safe(p("r_motion_vectors", fid))
        depth_np = load_npy_safe(p("r_depth", fid))
        jitter_np = load_npy_safe(p("r_jitter", fid))
        if any(x is None for x in [colour_np, motion_np, depth_np, jitter_np]):
            skipped += 1
            continue

        colour = to_chw(colour_np, channels=3, name="colour")
        motion = to_chw(motion_np, channels=2, name="motion")
        depth = to_chw(depth_np, channels=1, name="depth")
        jitter = to_jitter_c11(jitter_np)

        # Convert XY -> YX for NSS convention
        motion = xy_to_yx(motion, MOTION_INPUT_ORDER)
        jitter = xy_to_yx(jitter, JITTER_INPUT_ORDER)

        # Depth handling
        if REVERSE_Z:
            depth = 1.0 - depth

        colour_list.append(colour)
        motion_list.append(motion)
        depth_list.append(depth)
        jitter_list.append(jitter)

    if not colour_list:
        raise RuntimeError("❌ No valid frames after checking file completeness")
    if skipped > 0:
        print(f"⚠️ Skipped incomplete frames: {skipped}")

    colour_tensor = torch.stack(colour_list, dim=0) * float(COLOUR_SCALE)  # [T,3,H,W]
    motion_lr_tensor = torch.stack(motion_list, dim=0)  # [T,2,H,W], YX, pixels
    depth_tensor = torch.stack(depth_list, dim=0)  # [T,1,H,W]
    jitter_tensor = torch.stack(jitter_list, dim=0)  # [T,2,1,1], YX, pixels

    t, _, h, w = colour_tensor.shape
    out_h = int(round(h * SCALE))
    out_w = int(round(w * SCALE))

    # Build HR motion from LR motion (scale-aware)
    motion_hr_tensor = F.interpolate(
        motion_lr_tensor, size=(out_h, out_w), mode="nearest"
    )
    motion_hr_tensor[:, 0, ...] *= float(out_h) / float(h)
    motion_hr_tensor[:, 1, ...] *= float(out_w) / float(w)

    # Placeholder ground truth (for format compatibility)
    ground_truth_linear = F.interpolate(
        colour_tensor, size=(out_h, out_w), mode="bilinear", align_corners=False
    )

    render_size = torch.tensor([[h, w]], dtype=torch.int32).repeat(t, 1)
    out_dims = torch.tensor([[out_h, out_w]], dtype=torch.int32).repeat(t, 1)
    exposure = torch.zeros((t, 1), dtype=torch.float32)  # log exposure

    (
        z_near_t,
        z_far_t,
        fov_y_t,
        reverse_z_t,
        infinite_z_far_t,
        depth_params,
    ) = make_depth_params(
        render_size=render_size,
        z_near=Z_NEAR,
        z_far=Z_FAR,
        fov_y_rad=FOV_Y_RAD,
        reverse_z=REVERSE_Z,
        infinite_z_far=INFINITE_Z_FAR,
    )

    seq = torch.full((t, 1), int(SEQ_ID), dtype=torch.int64)
    scale_t = torch.full((t, 1), float(SCALE), dtype=torch.float32)

    print("📦 Final tensor shapes:")
    print(" colour_linear       :", tuple(colour_tensor.shape))
    print(" ground_truth_linear :", tuple(ground_truth_linear.shape))
    print(" motion_lr           :", tuple(motion_lr_tensor.shape))
    print(" motion              :", tuple(motion_hr_tensor.shape))
    print(" depth               :", tuple(depth_tensor.shape))
    print(" jitter              :", tuple(jitter_tensor.shape))
    print(" depth_params        :", tuple(depth_params.shape))

    tensors = {
        "colour_linear": colour_tensor.to(torch.float32),
        "ground_truth_linear": ground_truth_linear.to(torch.float32),
        "motion_lr": motion_lr_tensor.to(torch.float32),
        "motion": motion_hr_tensor.to(torch.float32),
        "depth": depth_tensor.to(torch.float32),
        "jitter": jitter_tensor.to(torch.float32),
        "depth_params": depth_params.to(torch.float32),
        "render_size": render_size,
        "outDims": out_dims,
        "exposure": exposure,
        "zNear": z_near_t.to(torch.float32),
        "zFar": z_far_t.to(torch.float32),
        "ReverseZ": reverse_z_t,
        "infinite_zFar": infinite_z_far_t,
        "FovY": fov_y_t.to(torch.float32),
        "scale": scale_t,
        "seq": seq,
    }

    metadata = {"Length": str(t)}
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)

    print("🎉 Saved NSS-compatible VNSS sequence to:")
    print(f"   {out_path}")


if __name__ == "__main__":
    main()

