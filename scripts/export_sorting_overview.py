#!/usr/bin/env python3
"""Render the complete Sorting table, robot and surrounding floor."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from custom_sorting import SortingEnv
from custom_sorting.task_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/figures/sorting_scene.png")
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--distance", type=float, default=2.9)
    parser.add_argument("--azimuth", type=float, default=125.0)
    parser.add_argument("--elevation", type=float, default=-28.0)
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.distance <= 0:
        parser.error("width, height and distance must be positive")

    cfg = load_config(ROOT / "configs/sorting.yaml")
    env = SortingEnv(config=cfg, target_object="red_cube", target_bin="blue_bin", seed=args.seed)
    try:
        env.reset(seed=args.seed)
        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(20):
            env.step(action)
        context = env.sim._render_context_offscreen
        context.cam.lookat[:] = [0.0, 0.0, cfg.table_offset[2] * 0.85]
        context.cam.distance = args.distance
        context.cam.azimuth = args.azimuth
        context.cam.elevation = args.elevation
        context.render(args.width, args.height, camera_id=-1)
        rgb = context.read_pixels(args.width, args.height)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not save {args.output}")
        print(f"Saved {args.output} ({args.width}x{args.height})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
