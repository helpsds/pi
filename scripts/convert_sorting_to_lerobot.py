#!/usr/bin/env python3
"""Convert successful raw sorting episodes into a LeRobot video dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lerobot/src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "outputs/sorting_raw_20"))
    parser.add_argument("--output", default=str(ROOT / "data/pi05_sorting_demo"))
    parser.add_argument("--repo-id", default="local/pi05_sorting_demo")
    return parser.parse_args()


def dataset_features(image_size: int):
    image = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.front": image,
        "observation.images.wrist": image.copy(),
        "observation.state": {"dtype": "float32", "shape": (9,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
        "grasp_success": {"dtype": "bool", "shape": (1,), "names": None},
        "place_success": {"dtype": "bool", "shape": (1,), "names": None},
    }


def main():
    args = parse_args()
    input_root, output_root = Path(args.input).resolve(), Path(args.output).resolve()
    if output_root.exists():
        raise FileExistsError(f"LeRobot root must not exist; refusing to overwrite: {output_root}")
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Raw manifest contains no successful episodes")
    with np.load(input_root / manifest[0]["file"], allow_pickle=False) as first:
        fps, image_size = int(first["fps"]), int(first["front"].shape[1])
    print(json.dumps({**vars(args), "episodes": len(manifest), "fps": fps, "image_size": image_size}, indent=2))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, root=output_root, robot_type="Panda", fps=fps,
        features=dataset_features(image_size), use_videos=True, image_writer_threads=4,
    )
    try:
        for expected_episode, record in enumerate(manifest):
            if record["episode"] != expected_episode:
                raise ValueError(f"Non-contiguous raw manifest at {record}")
            with np.load(input_root / record["file"], allow_pickle=False) as episode:
                length = len(episode["action"])
                if episode["action"].shape != (length, 7) or episode["state"].shape != (length, 9):
                    raise ValueError(f"Bad shapes in {record['file']}")
                if not bool(episode["done"][-1]) or not bool(episode["place_success"][-1]):
                    raise ValueError(f"Raw episode is not successful: {record['file']}")
                task = str(episode["task"].item())
                for idx in range(length):
                    dataset.add_frame({
                        "observation.images.front": episode["front"][idx],
                        "observation.images.wrist": episode["wrist"][idx],
                        "observation.state": episode["state"][idx], "action": episode["action"][idx],
                        "reward": np.asarray([episode["reward"][idx]], dtype=np.float32),
                        "done": np.asarray([episode["done"][idx]], dtype=np.bool_),
                        "grasp_success": np.asarray([episode["grasp_success"][idx]], dtype=np.bool_),
                        "place_success": np.asarray([episode["place_success"][idx]], dtype=np.bool_), "task": task,
                    })
            dataset.save_episode()
            print(f"ENCODED episode={expected_episode} frames={length} task={task}", flush=True)
    finally:
        dataset.finalize()
    print(f"LEROBOT_CONVERSION_COMPLETE episodes={len(manifest)} root={output_root}")


if __name__ == "__main__":
    main()
