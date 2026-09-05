#!/usr/bin/env python3
"""Collect successful expert rollouts to lossless raw episode files.

Run in the Python 3.11 ``robocasa`` environment where EGL is known to work.
Then convert with ``convert_sorting_to_lerobot.py`` in Python 3.12. Keeping
simulation and encoding separate avoids changing either working environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from custom_sorting import SortingEnv
from custom_sorting.expert import SortingExpert
from custom_sorting.task_config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sorting.yaml"))
    parser.add_argument("--output", default=str(ROOT / "outputs/sorting_raw_20"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite raw collection directory: {output}")
    output.mkdir(parents=True)
    print(json.dumps({**vars(args), "output": str(output), "resolved_config": vars(cfg)}, indent=2, default=list))

    successes, attempts, manifest = 0, 0, []
    while successes < args.episodes and attempts < args.max_attempts:
        seed = args.seed_start + attempts
        target_object, target_bin = cfg.task_combinations[successes % len(cfg.task_combinations)]
        env = SortingEnv(config=cfg, target_object=target_object, target_bin=target_bin, seed=seed)
        attempts += 1
        try:
            obs = env.reset(seed=seed)
            expert = SortingExpert(env)
            trajectory = {
                "front": [], "wrist": [], "state": [], "action": [],
                "reward": [], "done": [], "grasp_success": [], "place_success": [],
            }
            final_metrics = env.success_metrics()
            for step_idx in range(args.max_steps):
                expert_step = expert.act()
                action = expert_step.action
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise RuntimeError(f"Invalid expert action at seed={seed}, step={step_idx}: {action}")
                next_obs, reward, done, info = env.step(action)
                trajectory["front"].append(np.flipud(obs["agentview_image"]).copy())
                trajectory["wrist"].append(np.flipud(obs["robot0_eye_in_hand_image"]).copy())
                trajectory["state"].append(np.asarray(obs["sorting_state"], dtype=np.float32))
                trajectory["action"].append(np.asarray(action, dtype=np.float32))
                trajectory["reward"].append(np.float32(reward))
                trajectory["done"].append(np.bool_(info["task_success"]))
                trajectory["grasp_success"].append(np.bool_(info["grasp_success"]))
                trajectory["place_success"].append(np.bool_(info["place_success"]))
                obs, final_metrics = next_obs, info
                if info["task_success"] or done:
                    break
            if not final_metrics["task_success"]:
                print(f"DISCARDED seed={seed} final_metrics={final_metrics}", flush=True)
                continue

            filename = f"episode_{successes:06d}.npz"
            np.savez_compressed(
                output / filename,
                **{key: np.stack(values) for key, values in trajectory.items()},
                task=np.asarray(env.instruction), seed=np.asarray(seed, dtype=np.int64),
                target_object=np.asarray(target_object), target_bin=np.asarray(target_bin),
                fps=np.asarray(cfg.control_freq, dtype=np.int64),
            )
            record = {
                "episode": successes, "seed": seed, "task": env.instruction,
                "target_object": target_object, "target_bin": target_bin,
                "frames": step_idx + 1, "file": filename,
            }
            manifest.append(record)
            print("COLLECTED=" + json.dumps(record, sort_keys=True), flush=True)
            successes += 1
        finally:
            env.close()

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if successes != args.episodes:
        raise RuntimeError(f"Collected only {successes}/{args.episodes} episodes in {attempts} attempts")
    print(f"RAW_COLLECTION_COMPLETE episodes={successes} attempts={attempts} root={output}")


if __name__ == "__main__":
    main()
