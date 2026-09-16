# Environment setup

The recorded experiment used Ubuntu/Linux, NVIDIA driver 570.133.07 and four RTX 4090 GPUs (24 GB each). Training used GPUs 0–3; evaluation used two inference GPUs and two EGL GPUs.

## Tested versions

| Component | Version / commit |
|---|---|
| Python | 3.12.14 (training and policy client) |
| Python | 3.11 (MuJoCo environment server) |
| PyTorch | 2.7.1+cu128 |
| Transformers | 5.5.4 |
| LeRobot | 0.6.2, commit `4aaff99be4a1d81568c08c8f0296b41b40c99ec4` |
| MuJoCo | 3.8.1 |
| robosuite | commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| RoboCasa | commit `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` |
| NumPy / SciPy | 2.2.6 / 1.18.1 |

Exact commits are recorded because robotics APIs and controller semantics change frequently. If a future LeRobot release changes π0.5 configuration fields, use the recorded commit first.

## Clone dependencies

```bash
git clone https://github.com/helpsds/pi.git
cd pi

git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout 4aaff99be4a1d81568c08c8f0296b41b40c99ec4

mkdir -p third_party
git clone https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite
git -C third_party/robosuite checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git clone https://github.com/robocasa/robocasa.git third_party/robocasa
git -C third_party/robocasa checkout a07e365c958c4216cd6bbd5f30b47f09a65c6f00
```

The 30k π0.5 run also requires this repository's [three-file LeRobot patch](../patches/lerobot_pi05_visual_tokenizer.patch). Apply it to the pinned, clean LeRobot checkout before training:

```bash
git -C lerobot apply --check "$PWD/patches/lerobot_pi05_visual_tokenizer.patch"
git -C lerobot apply "$PWD/patches/lerobot_pi05_visual_tokenizer.patch"
```

It adds the `train_vision_with_expert` option and a local tokenizer path override; it does not replace action transforms. See the [README](../README.md) for verification and already-applied handling.

## Policy environment (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './lerobot[training,pi]'
python -m pip install numpy scipy opencv-python matplotlib pyyaml pyarrow
```

Install the PyTorch build appropriate for your CUDA driver. The recorded run used `torch==2.7.1+cu128`.

Download π0.5 Base according to the current LeRobot/Hugging Face model instructions. Some PaliGemma assets are gated; accept their license and authenticate with `hf auth login`. Never commit the token.

## Simulation environment (Python 3.11)

Keeping MuJoCo/robosuite separate from the policy environment avoids OpenGL and binary dependency conflicts.

```bash
conda create -n robocasa python=3.11 -y
conda activate robocasa
pip install -e third_party/robosuite
pip install -e third_party/robocasa
pip install mujoco==3.8.1 opencv-python pyyaml
```

## EGL smoke test

```bash
conda activate robocasa
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
PYTHONPATH=$PWD/third_party/robosuite:$PWD \
python scripts/collect_sorting_demos.py --episodes 1 --max-attempts 2 \
  --output /tmp/sorting_egl_smoke
```

`MUJOCO_EGL_DEVICE_ID` is a physical ID within `CUDA_VISIBLE_DEVICES`. A mismatch causes an assertion before environment creation.
