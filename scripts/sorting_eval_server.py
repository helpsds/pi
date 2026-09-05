#!/usr/bin/env python3
"""Python 3.11/EGL sorting environment server for external policy clients."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from custom_sorting import SortingEnv
from custom_sorting.task_config import load_config
from wire_protocol import recv_obj, send_obj


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sorting.yaml"))
    parser.add_argument("--target-object", default="red_cube")
    parser.add_argument("--target-bin", default="blue_bin")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--video", default=str(ROOT / "outputs/pi05_sorting_base_smoke.mp4"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(json.dumps({**vars(args), "resolved_config": vars(cfg)}, indent=2, default=list), flush=True)
    env = SortingEnv(
        config=cfg, target_object=args.target_object, target_bin=args.target_bin,
        seed=args.seed, horizon=args.horizon,
    )
    obs = env.reset(seed=args.seed)
    video_path = Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.control_freq,
        (cfg.image_size, cfg.image_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Waiting for policy client on {args.host}:{args.port}", flush=True)
    conn = None
    try:
        conn, address = server.accept()
        print(f"Policy connected: {address}", flush=True)
        final_metrics = env.success_metrics()
        clipped_steps = 0
        for step_idx in range(args.horizon):
            front = np.flipud(obs["agentview_image"]).copy()
            wrist = np.flipud(obs["robot0_eye_in_hand_image"]).copy()
            writer.write(cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
            send_obj(conn, {
                "type": "observation", "step": step_idx, "task": env.instruction,
                "observation": {
                    "observation.images.front": front,
                    "observation.images.wrist": wrist,
                    "observation.state": np.asarray(obs["sorting_state"], dtype=np.float32),
                },
            })
            reply = recv_obj(conn)
            if reply.get("type") != "action":
                raise RuntimeError(f"Expected action message, got {reply}")
            raw_action = np.asarray(reply["action"], dtype=np.float32)
            if raw_action.shape != (7,):
                raise ValueError(f"Expected 7D action, got {raw_action.shape}")
            if not np.isfinite(raw_action).all():
                raise ValueError(f"Non-finite policy action: {raw_action}")
            action = np.clip(raw_action, -1.0, 1.0)
            clipped = not np.array_equal(action, raw_action)
            clipped_steps += int(clipped)
            obs, reward, done, final_metrics = env.step(action)
            if step_idx < 10 or step_idx % 20 == 0:
                print(
                    f"step={step_idx:03d} reward={reward:.1f} clipped={clipped} "
                    f"action={np.array2string(raw_action, precision=3)} metrics={final_metrics}",
                    flush=True,
                )
            if final_metrics["task_success"] or done:
                break
        result = {
            "type": "result", "success": bool(final_metrics["task_success"]),
            "grasp_success": bool(final_metrics["grasp_success"]),
            "place_success": bool(final_metrics["place_success"]),
            "steps": step_idx + 1, "seed": args.seed, "task": env.instruction,
            "clipped_steps": clipped_steps, "video": str(video_path.resolve()),
        }
        send_obj(conn, result)
        print("SORTING_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    finally:
        if conn is not None:
            conn.close()
        server.close()
        writer.release()
        env.close()


if __name__ == "__main__":
    main()
