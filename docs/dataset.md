# Dataset pipeline

## Final training dataset

| Property | Value |
|---|---:|
| LeRobot version | v3.0 |
| Episodes | 200 |
| Frames | 52,118 |
| Task labels | 18 |
| FPS metadata | 10 |
| Images | front + wrist, RGB 256×256 |
| State | 8D |
| Action | 7D |

State layout:

```text
[eef_x, eef_y, eef_z, rotvec_x, rotvec_y, rotvec_z, finger_1, finger_2]
```

Action layout:

```text
[delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
```

The Sorting simulator originally exposes a 9D state containing an `xyzw` quaternion. `build_sorting_mimicgen_combined.py` converts it to a three-dimensional axis-angle rotation vector so both sources share an 8D state.

## 1. Collect Sorting demonstrations

Collect 80 successful trajectories: eight object/bin commands, ten demonstrations each.

```bash
conda activate robocasa
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
PYTHONPATH=$PWD/third_party/robosuite:$PWD \
python scripts/collect_sorting_demos.py \
  --config configs/sorting.yaml \
  --episodes 80 --max-attempts 200 \
  --output outputs/sorting_raw_8tasks_x10
```

Only successful expert episodes are saved. Raw episodes are compressed NPZ files plus `manifest.json`.

## 2. Prepare MimicGen

Download and preserve the original LeRobot v2.1 dataset:

```text
https://huggingface.co/datasets/yananchen/mimicgen_source_12tasks_v21
```

Convert a copy to LeRobot v3.0 using the conversion command provided by the installed LeRobot version. Keep the original directory unchanged. The expected converted root is:

```text
data/external/mimicgen_source_12tasks_v30_work
```

The public repository deliberately does not redistribute this dataset. Dataset licensing and access remain with its publisher.

## 3. Build the harmonized 200-episode dataset

```bash
source .venv/bin/activate
PYTHONPATH=$PWD/lerobot/src:$PWD \
python scripts/build_sorting_mimicgen_combined.py \
  --sorting-raw outputs/sorting_raw_8tasks_x10 \
  --mimicgen-root data/external/mimicgen_source_12tasks_v30_work \
  --output data/pi05_sorting_mimicgen_200eps \
  --repo-id local/pi05_sorting_mimicgen_200eps
```

The conversion performs these checks:

- Sorting quaternion state must be exactly 9D before conversion.
- Every merged state must be 8D and every action 7D.
- Both image streams are converted to uint8 HWC and resized to 256×256.
- The destination is finalized only after both sources are processed.

## Timing limitation

Sorting was collected at 20 Hz while the MimicGen source uses 10 Hz. The completed experiment stored a dataset-level FPS of 10 and did not drop or duplicate Sorting frames. This is a known limitation and a useful future ablation: resample trajectories before merging and compare policy performance.
