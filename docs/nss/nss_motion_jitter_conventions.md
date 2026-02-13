# NSS Motion/Jitter Conventions (Internal Safetensors Path)

This note documents the conventions used by the NSS data pipeline in this repository,
from EXR/JSON capture data into the tensors consumed by training/inference.

## 1) Motion sign

- Internal warping uses `query = grid - flow` (`dense_image_warp`), so motion is interpreted as:
  `flow_t = p_t - p_(t-1)` (current minus previous).
- Practical meaning:
  - `+X` means moving right on screen.
  - `+Y` means moving down on screen.
- The writer does **not** flip motion sign; it only reorders channels and rescales units.

Relevant code:
- `src/ng_model_gym/core/model/dense_warp_utils.py` (`dense_image_warp`)
- `scripts/safetensors_generator/dataset_reader.py` (`process_motion`)

## 2) Motion units

- In capture format (`motion` / `motion_gt` EXR), vectors are in normalized UV space.
- During EXR -> safetensors conversion, motion is converted to **pixel units**:
  - channel 0 multiplied by image height
  - channel 1 multiplied by image width
- For shader-accurate path, `motion_lr` is normalized again in dataloader using
  `fixed_normalize_mvs(..., height=544, width=960)`.

Relevant code:
- `scripts/safetensors_generator/dataset_reader.py` (`process_motion`)
- `src/ng_model_gym/usecases/nss/data/dataset.py` (`fixed_normalize_mvs`)

## 3) Motion channel order (XY or YX)

- EXR input is read as `RG` (engine-style XY / UV order).
- Writer explicitly swizzles with `"yx"` and stores motion internally as **YX**:
  - channel 0: Y (vertical / v)
  - channel 1: X (horizontal / u)

Relevant code:
- `scripts/safetensors_generator/dataset_reader.py` (`swizzle(motion, "yx")`)

## 4) Jitter channel order (XY or YX)

- Metadata provides jitter as `X`, `Y`.
- Writer stores jitter tensor as **YX** (same convention as motion):
  - `jitter = [Y, X]`
- Jitter is stored in **pixel units** internally.

Relevant code:
- `scripts/safetensors_generator/dataset_reader.py`

## 5) Jitter centering

- Expected centered jitter domain is around pixel center offset `[-0.5, 0.5]` in pixel space.
- Metadata field `NormalizedPerRatioJitter` is converted to pixels by multiplying with source
  render size; this reconstructs centered pixel jitter.
- Jitter-aware LUT math uses `jitter + 0.5`, which assumes centered jitter input.

Relevant code:
- `docs/nss/nss_dataset_specification.md`
- `scripts/safetensors_generator/dataset_reader.py`
- `src/ng_model_gym/core/model/graphics_utils.py` (`generate_lr_to_hr_lut`, `compute_jitter_tile_offset`)
