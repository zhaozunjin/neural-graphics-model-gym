#!/usr/bin/env python3
# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""Convert NSS-style NPY tensors into a single .safetensors sequence file.

This helper is intended for cases where data has already been captured as NPY arrays
instead of EXR files expected by the existing safetensors_writer pipeline.

Input arrays can be channel-first (T,C,H,W) or channel-last (T,H,W,C) for colour/motion/depth.
Jitter supports (T,2), (T,2,1,1), or (T,1,1,2).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from scripts.safetensors_generator.fsr2_methods import depth_to_view_space_params


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def _to_tchw(array: np.ndarray, channels: int, name: str) -> torch.Tensor:
    """Convert an array into (T, C, H, W)."""
    arr = np.asarray(array)

    if arr.ndim == 4:
        if arr.shape[1] == channels:
            out = arr
        elif arr.shape[-1] == channels:
            out = np.transpose(arr, (0, 3, 1, 2))
        else:
            raise ValueError(
                f"{name} has unsupported shape {arr.shape}. "
                f"Expected channel dim == {channels}."
            )
    elif arr.ndim == 3 and channels == 1:
        # (T, H, W) -> (T, 1, H, W)
        out = arr[:, None, :, :]
    else:
        raise ValueError(
            f"{name} has unsupported shape {arr.shape}. "
            "Expected (T,C,H,W), (T,H,W,C), or (T,H,W) for 1-channel tensors."
        )

    return torch.from_numpy(out).to(torch.float32)


def _to_jitter_tchw(array: np.ndarray) -> torch.Tensor:
    """Convert jitter into (T, 2, 1, 1)."""
    arr = np.asarray(array)

    if arr.ndim == 2 and arr.shape[1] == 2:
        out = arr[:, :, None, None]
    elif arr.ndim == 4:
        if arr.shape[1] == 2 and arr.shape[2] == 1 and arr.shape[3] == 1:
            out = arr
        elif arr.shape[1] == 1 and arr.shape[2] == 1 and arr.shape[3] == 2:
            out = np.transpose(arr, (0, 3, 1, 2))
        else:
            raise ValueError(
                f"Unsupported jitter shape {arr.shape}. "
                "Expected (T,2,1,1) or (T,1,1,2) when ndim=4."
            )
    else:
        raise ValueError(
            f"Unsupported jitter shape {arr.shape}. "
            "Expected (T,2), (T,2,1,1), or (T,1,1,2)."
        )

    return torch.from_numpy(out).to(torch.float32)


def _check_frames(name: str, t_expected: int, tensor: torch.Tensor) -> None:
    if tensor.shape[0] != t_expected:
        raise ValueError(
            f"{name} has T={tensor.shape[0]} but expected T={t_expected}."
        )


def _to_yx(vec: torch.Tensor, order: str) -> torch.Tensor:
    """Ensure vector channels are (Y, X)."""
    if order == "yx":
        return vec
    if order == "xy":
        return vec[:, [1, 0], ...]
    raise ValueError(f"Unsupported vector order: {order}")


def _uv_to_pixels_yx(vec: torch.Tensor, height: int, width: int) -> torch.Tensor:
    out = vec.clone()
    out[:, 0, ...] *= float(height)
    out[:, 1, ...] *= float(width)
    return out


def _make_depth_params(
    render_size: torch.Tensor,
    z_near: float,
    z_far: float,
    fov_y_rad: float,
    reverse_z: bool,
    infinite_z_far: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create NSS depth-related scalar tensors and depth_params."""
    t = render_size.shape[0]
    device = render_size.device

    z_near_t = torch.full((t, 1), float(z_near), dtype=torch.float32, device=device)
    z_far_t = torch.full((t, 1), float(z_far), dtype=torch.float32, device=device)
    fov_y_t = torch.full((t, 1), float(fov_y_rad), dtype=torch.float32, device=device)
    reverse_z_t = torch.full((t, 1), bool(reverse_z), dtype=torch.bool, device=device)
    infinite_z_far_t = torch.full(
        (t, 1), bool(infinite_z_far), dtype=torch.bool, device=device
    )

    make_image_like = lambda x: x.unsqueeze(-1).unsqueeze(-1)
    depth_params = depth_to_view_space_params(
        zNear=make_image_like(z_near_t),
        zFar=make_image_like(z_far_t),
        FovY=make_image_like(fov_y_t),
        infinite=make_image_like(infinite_z_far_t),
        renderSizeWidth=make_image_like(render_size[:, 1:2].to(torch.float32)),
        renderSizeHeight=make_image_like(render_size[:, 0:1].to(torch.float32)),
        inverted=reverse_z_t,
    ).squeeze(-1).squeeze(-1)

    return z_near_t, z_far_t, fov_y_t, reverse_z_t, infinite_z_far_t, depth_params


def _prepare_ground_truth(
    colour_linear: torch.Tensor,
    out_height: int,
    out_width: int,
    ground_truth_npy: Path | None,
    dummy_ground_truth: str,
) -> torch.Tensor:
    if ground_truth_npy is not None:
        gt = _to_tchw(_load_npy(ground_truth_npy), channels=3, name="ground_truth")
        _check_frames("ground_truth", colour_linear.shape[0], gt)
        if gt.shape[2] != out_height or gt.shape[3] != out_width:
            gt = F.interpolate(
                gt, size=(out_height, out_width), mode="bilinear", align_corners=False
            )
        return gt

    if dummy_ground_truth == "upsample_color":
        return F.interpolate(
            colour_linear,
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
        )

    if dummy_ground_truth == "copy_color":
        if colour_linear.shape[2] != out_height or colour_linear.shape[3] != out_width:
            raise ValueError(
                "dummy_ground_truth='copy_color' requires output dims == input dims."
            )
        return colour_linear.clone()

    raise ValueError(f"Unsupported dummy_ground_truth option: {dummy_ground_truth}")


def _prepare_exposure(
    t: int, exposure_npy: Path | None, exposure_is_linear: bool
) -> torch.Tensor:
    if exposure_npy is None:
        return torch.zeros((t, 1), dtype=torch.float32)

    arr = np.asarray(_load_npy(exposure_npy))
    if arr.ndim == 1 and arr.shape[0] == t:
        exp = torch.from_numpy(arr[:, None]).to(torch.float32)
    elif arr.ndim == 2 and arr.shape == (t, 1):
        exp = torch.from_numpy(arr).to(torch.float32)
    else:
        raise ValueError(
            f"Unsupported exposure shape {arr.shape}. Expected (T,) or (T,1)."
        )

    if exposure_is_linear:
        exp = torch.log(torch.clamp(exp, min=1e-6))
    return exp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NSS NPY tensors to a single .safetensors sequence."
    )
    parser.add_argument("--color-npy", type=Path, required=True)
    parser.add_argument("--jitter-npy", type=Path, required=True)
    parser.add_argument("--motion-npy", type=Path, required=True)
    parser.add_argument("--depth-npy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--ground-truth-npy",
        type=Path,
        default=None,
        help="Optional ground-truth color array.",
    )
    parser.add_argument(
        "--dummy-ground-truth",
        type=str,
        choices=["upsample_color", "copy_color"],
        default="upsample_color",
        help="How to synthesize ground_truth_linear when --ground-truth-npy is missing.",
    )

    parser.add_argument(
        "--motion-resolution",
        type=str,
        choices=["lr", "hr"],
        default="lr",
        help=(
            "Whether --motion-npy is at input (lr) or output (hr) resolution. "
            "This controls how motion/motion_lr are derived."
        ),
    )
    parser.add_argument(
        "--motion-order",
        type=str,
        choices=["yx", "xy"],
        default="yx",
        help="Channel order in motion file.",
    )
    parser.add_argument(
        "--jitter-order",
        type=str,
        choices=["yx", "xy"],
        default="yx",
        help="Channel order in jitter file.",
    )
    parser.add_argument(
        "--motion-units",
        type=str,
        choices=["pixels", "uv"],
        default="pixels",
        help="Units used by motion file.",
    )
    parser.add_argument(
        "--jitter-units",
        type=str,
        choices=["pixels", "uv"],
        default="pixels",
        help="Units used by jitter file.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Output/input ratio used when output dims are not explicitly provided.",
    )
    parser.add_argument("--out-height", type=int, default=None)
    parser.add_argument("--out-width", type=int, default=None)

    parser.add_argument("--z-near", type=float, default=0.1)
    parser.add_argument("--z-far", type=float, default=1000.0)
    parser.add_argument("--fov-y-rad", type=float, default=1.0471975512)  # 60 deg
    parser.add_argument("--reverse-z", action="store_true", default=False)
    parser.add_argument("--infinite-z-far", action="store_true", default=False)
    parser.add_argument("--sequence-id", type=int, default=1)

    parser.add_argument("--exposure-npy", type=Path, default=None)
    parser.add_argument(
        "--exposure-is-linear",
        action="store_true",
        default=False,
        help="If set, exposure values are converted with log() before writing.",
    )

    args = parser.parse_args()

    colour_linear = _to_tchw(_load_npy(args.color_npy), channels=3, name="colour_linear")
    depth = _to_tchw(_load_npy(args.depth_npy), channels=1, name="depth")
    jitter = _to_jitter_tchw(_load_npy(args.jitter_npy))
    motion_in = _to_tchw(_load_npy(args.motion_npy), channels=2, name="motion")

    t, _, in_height, in_width = colour_linear.shape
    _check_frames("depth", t, depth)
    _check_frames("jitter", t, jitter)
    _check_frames("motion", t, motion_in)

    if depth.shape[2] != in_height or depth.shape[3] != in_width:
        raise ValueError(
            "depth resolution must match color input resolution. "
            f"Got depth={tuple(depth.shape[2:])}, color={tuple(colour_linear.shape[2:])}."
        )

    if args.out_height is not None or args.out_width is not None:
        if args.out_height is None or args.out_width is None:
            raise ValueError("Both --out-height and --out-width must be provided together.")
        out_height = int(args.out_height)
        out_width = int(args.out_width)
    else:
        out_height = int(round(in_height * float(args.scale)))
        out_width = int(round(in_width * float(args.scale)))

    if out_height <= 0 or out_width <= 0:
        raise ValueError(f"Invalid output dimensions: ({out_height}, {out_width})")

    scale_y = float(out_height) / float(in_height)
    scale_x = float(out_width) / float(in_width)

    motion_in = _to_yx(motion_in, args.motion_order)
    jitter = _to_yx(jitter, args.jitter_order)

    if args.motion_units == "uv":
        motion_in = _uv_to_pixels_yx(motion_in, in_height, in_width)
    if args.jitter_units == "uv":
        jitter = _uv_to_pixels_yx(jitter, in_height, in_width)

    if args.motion_resolution == "lr":
        motion_lr = motion_in
        motion = F.interpolate(motion_lr, size=(out_height, out_width), mode="nearest")
        motion[:, 0, ...] *= scale_y
        motion[:, 1, ...] *= scale_x
    else:
        motion = motion_in
        if motion.shape[2] != out_height or motion.shape[3] != out_width:
            raise ValueError(
                "With --motion-resolution hr, motion tensor resolution must match output dims. "
                f"Got motion={tuple(motion.shape[2:])}, out=({out_height}, {out_width})."
            )
        motion_lr = F.interpolate(motion, size=(in_height, in_width), mode="nearest")
        motion_lr[:, 0, ...] *= float(in_height) / float(out_height)
        motion_lr[:, 1, ...] *= float(in_width) / float(out_width)

    ground_truth_linear = _prepare_ground_truth(
        colour_linear,
        out_height=out_height,
        out_width=out_width,
        ground_truth_npy=args.ground_truth_npy,
        dummy_ground_truth=args.dummy_ground_truth,
    )
    _check_frames("ground_truth_linear", t, ground_truth_linear)

    exposure = _prepare_exposure(
        t=t, exposure_npy=args.exposure_npy, exposure_is_linear=args.exposure_is_linear
    )

    render_size = torch.tensor([[in_height, in_width]], dtype=torch.int32).repeat(t, 1)
    out_dims = torch.tensor([[out_height, out_width]], dtype=torch.int32).repeat(t, 1)

    (
        z_near_t,
        z_far_t,
        fov_y_t,
        reverse_z_t,
        infinite_z_far_t,
        depth_params,
    ) = _make_depth_params(
        render_size=render_size,
        z_near=args.z_near,
        z_far=args.z_far,
        fov_y_rad=args.fov_y_rad,
        reverse_z=args.reverse_z,
        infinite_z_far=args.infinite_z_far,
    )

    aspect = float(in_width) / float(in_height)
    fov_x_value = 2.0 * np.arctan(np.tan(float(args.fov_y_rad) * 0.5) * aspect)
    fov_x_t = torch.full((t, 1), float(fov_x_value), dtype=torch.float32)

    seq = torch.full((t, 1), int(args.sequence_id), dtype=torch.int64)
    img = torch.arange(t, dtype=torch.int64).view(t, 1)

    scale_value = float(out_width) / float(in_width)
    scale_t = torch.full((t, 1), scale_value, dtype=torch.float32)

    # Optional metadata-compatible fields used by some tooling.
    jitter_norm_y = jitter[:, 0, 0, 0] / float(in_height)
    jitter_norm_x = jitter[:, 1, 0, 0] / float(in_width)
    x = jitter_norm_x.view(t, 1)
    y = (-jitter_norm_y).view(t, 1)
    view_proj = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(t, 1, 1, 1)

    output_tensors = {
        "colour_linear": colour_linear.to(torch.float32),
        "ground_truth_linear": ground_truth_linear.to(torch.float32),
        "motion": motion.to(torch.float32),
        "motion_lr": motion_lr.to(torch.float32),
        "depth": depth.to(torch.float32),
        "depth_params": depth_params.to(torch.float32),
        "exposure": exposure.to(torch.float32),
        "jitter": jitter.to(torch.float32),
        "render_size": render_size,
        "outDims": out_dims,
        "zNear": z_near_t.to(torch.float32),
        "zFar": z_far_t.to(torch.float32),
        "ReverseZ": reverse_z_t,
        "infinite_zFar": infinite_z_far_t,
        "FovX": fov_x_t,
        "FovY": fov_y_t.to(torch.float32),
        "seq": seq,
        "img": img,
        "scale": scale_t,
        "viewProj": view_proj,
        "X": x.to(torch.float32),
        "Y": y.to(torch.float32),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Length": str(t),
        "Created": datetime.now().strftime("%d-%m-%Y"),
        "Version": "NPYToSafetensorsV1",
    }
    save_file(output_tensors, args.output, metadata=metadata)

    print(f"Saved: {args.output}")
    print(f"Frames: {t}")
    print(f"Input resolution: {in_width}x{in_height}")
    print(f"Output resolution: {out_width}x{out_height}")
    print("Keys:", ", ".join(sorted(output_tensors.keys())))


if __name__ == "__main__":
    main()
