#!/usr/bin/env python3
"""Build a harmonized LeRobot v3 dataset from Sorting and MimicGen v2.1-derived data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lerobot/src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sorting-raw", default=str(ROOT / "outputs/sorting_raw_8tasks_x10"))
    p.add_argument(
        "--mimicgen-root", default=str(ROOT / "data/external/mimicgen_source_12tasks_v30_work")
    )
    p.add_argument("--output", default=str(ROOT / "data/pi05_sorting_mimicgen_200eps"))
    p.add_argument("--repo-id", default="local/pi05_sorting_mimicgen_200eps")
    return p.parse_args()


def features():
    image = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.front": image,
        "observation.images.wrist": image.copy(),
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "finger_1", "finger_2"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        },
    }


def uint8_hwc(tensor, resize=False):
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if resize or image.shape[:2] != (256, 256):
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return image


def sorting_state_to_axis_angle(state):
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (9,):
        raise ValueError(f"Expected Sorting 9D state, got {state.shape}")
    rotvec = Rotation.from_quat(state[3:7]).as_rotvec().astype(np.float32)
    return np.concatenate((state[:3], rotvec, state[7:9])).astype(np.float32)


def add_source(destination, source, kind):
    current_episode = None
    written_episodes = 0
    for index in range(len(source)):
        row = source[index]
        episode = int(row["episode_index"])
        if current_episode is not None and episode != current_episode:
            destination.save_episode()
            written_episodes += 1
        current_episode = episode
        if kind == "sorting":
            front = uint8_hwc(row["observation.images.front"])
            wrist = uint8_hwc(row["observation.images.wrist"])
            state = sorting_state_to_axis_angle(row["observation.state"])
            action = row["action"].detach().cpu().numpy().astype(np.float32)
        else:
            front = uint8_hwc(row["image"], resize=True)
            wrist = uint8_hwc(row["wrist_image"], resize=True)
            state = row["state"].detach().cpu().numpy().astype(np.float32)
            action = row["actions"].detach().cpu().numpy().astype(np.float32)
        if state.shape != (8,) or action.shape != (7,):
            raise ValueError(f"Bad {kind} shapes at {index}: state={state.shape}, action={action.shape}")
        destination.add_frame(
            {
                "observation.images.front": front,
                "observation.images.wrist": wrist,
                "observation.state": state,
                "action": action,
                "task": row["task"],
            }
        )
        if index % 2000 == 0:
            print(f"{kind}: frame {index}/{len(source)}", flush=True)
    if current_episode is not None:
        destination.save_episode()
        written_episodes += 1
    return written_episodes


def add_sorting_raw(destination, root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for episode_index, record in enumerate(manifest):
        with np.load(root / record["file"], allow_pickle=False) as episode:
            front = episode["front"]
            wrist = episode["wrist"]
            state = episode["state"]
            action = episode["action"]
            task = str(episode["task"].item())
            length = len(action)
            for index in range(length):
                destination.add_frame(
                    {
                        "observation.images.front": front[index],
                        "observation.images.wrist": wrist[index],
                        "observation.state": sorting_state_to_axis_angle(state[index]),
                        "action": action[index].astype(np.float32),
                        "task": task,
                    }
                )
        destination.save_episode()
        if episode_index % 10 == 0:
            print(f"sorting: episode {episode_index}/{len(manifest)}", flush=True)
    return len(manifest)


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    mimicgen = LeRobotDataset("local/mimicgen_source_12tasks_v30", root=args.mimicgen_root)
    destination = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        robot_type="Panda",
        fps=10,
        features=features(),
        use_videos=False,
        image_writer_threads=8,
    )
    try:
        sorting_episodes = add_sorting_raw(destination, args.sorting_raw)
        mimicgen_episodes = add_source(destination, mimicgen, "mimicgen")
    finally:
        destination.finalize()
    manifest = {
        "repo_id": args.repo_id,
        "output_version": "v3.0",
        "output_fps": 10,
        "state": "EEF position xyz + axis-angle rotation vector + two gripper joints",
        "action": "normalized delta EEF position/rotation + gripper command",
        "sources": {
            "sorting": {
                "root": str(Path(args.sorting_raw).resolve()),
                "source_version": "v3.0",
                "source_fps": 20,
                "episodes": sorting_episodes,
                "state_conversion": "xyzw quaternion to axis-angle rotation vector",
            },
            "mimicgen": {
                "original_v21": str((ROOT / "data/external/mimicgen_source_12tasks_v21").resolve()),
                "converted_v30": str(Path(args.mimicgen_root).resolve()),
                "source_fps": mimicgen.meta.fps,
                "episodes": mimicgen_episodes,
                "image_conversion": "84x84 to 256x256 INTER_AREA",
            },
        },
        "known_limitation": "Sorting source is 20Hz while MimicGen is 10Hz; frames/actions were not dropped or duplicated.",
    }
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("COMBINED_DATASET=" + json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
