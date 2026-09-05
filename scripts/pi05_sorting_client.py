#!/usr/bin/env python3
"""Python 3.12 LeRobot π0.5 client for the custom sorting server."""

from __future__ import annotations

import argparse
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lerobot/src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lerobot.configs import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from wire_protocol import recv_obj, send_obj


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", default=str(ROOT / "data/pi05_sorting_demo"))
    parser.add_argument("--repo-id", default="local/pi05_sorting_demo")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--policy-seed", type=int, required=True)
    return parser.parse_args()


def seed_policy(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_tensor(image):
    image = np.asarray(image)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HWC 256 image, got {image.shape} {image.dtype}")
    return torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous().float() / 255.0


def policy_state(state, expected_dim: int):
    """Match the environment's 9D xyzw-quaternion state to the training schema."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid Sorting environment state: {state}")
    if expected_dim == 9:
        return torch.from_numpy(state)
    if expected_dim == 8:
        rotvec = Rotation.from_quat(state[3:7]).as_rotvec().astype(np.float32)
        return torch.from_numpy(np.concatenate((state[:3], rotvec, state[7:9])).astype(np.float32))
    raise ValueError(f"Unsupported policy state dimension: {expected_dim}")


def main():
    args = parse_args()
    seed_policy(args.policy_seed)
    print(f"Policy RNG seed: {args.policy_seed}", flush=True)
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, download_videos=False)
    print(
        f"Dataset metadata: cameras={ds.meta.camera_keys} "
        f"state={ds.meta.features['observation.state']['shape']} "
        f"action={ds.meta.features['action']['shape']}", flush=True,
    )
    cfg = PI05Config(
        pretrained_path=args.model, device="cuda", dtype="bfloat16",
        n_action_steps=args.n_action_steps, empty_cameras=1,
        use_relative_actions=False,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
    )
    policy = make_policy(cfg, ds_meta=ds.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=ds.meta.stats, dataset_meta=ds.meta,
    )
    policy.eval()
    policy.reset()
    print(f"π0.5 loaded; output feature={cfg.output_features['action']}", flush=True)

    sock = socket.create_connection((args.host, args.port), timeout=30)
    sock.settimeout(None)
    print(f"Connected to {args.host}:{args.port}", flush=True)
    try:
        while True:
            msg = recv_obj(sock)
            if msg["type"] == "result":
                print(f"RESULT={msg}", flush=True)
                break
            if msg["type"] != "observation":
                raise RuntimeError(f"Unexpected message type: {msg['type']}")
            obs = msg["observation"]
            expected_state_dim = ds.meta.features["observation.state"]["shape"][0]
            raw = {
                "observation.images.front": image_tensor(obs["observation.images.front"]),
                "observation.images.wrist": image_tensor(obs["observation.images.wrist"]),
                "observation.state": policy_state(obs["observation.state"], expected_state_dim),
                "task": msg["task"],
            }
            policy_input = preprocessor(raw)
            with torch.inference_mode():
                action = postprocessor(policy.select_action(policy_input))
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"π0.5 returned invalid action: shape={action.shape} value={action}")
            if msg["step"] < 10 or msg["step"] % 20 == 0:
                print(f"step={msg['step']:03d} action={np.array2string(action, precision=3)}", flush=True)
            send_obj(sock, {"type": "action", "action": action})
    finally:
        sock.close()


if __name__ == "__main__":
    main()
