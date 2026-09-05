#!/usr/bin/env python3
"""Inspect a sorting LeRobot dataset and export one episode to MP4."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lerobot/src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "data/pi05_sorting_demo"))
    parser.add_argument("--repo-id", default="local/pi05_sorting_demo")
    parser.add_argument("--episode", type=int, default=None, help="Episode to export; default is seeded random choice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--video", default=str(ROOT / "outputs/sorting_dataset_inspect.mp4"))
    return parser.parse_args()


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def image_uint8(value):
    image = as_numpy(value)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255)
    return image.astype(np.uint8)


def main():
    args = parse_args()
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, download_videos=False)
    meta = ds.meta
    print("DATASET_FEATURES=" + json.dumps(meta.features, indent=2, default=list))

    parquet_files = sorted(glob.glob(str(Path(args.root) / "data/**/*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No data parquet files below {args.root}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=["action", "episode_index", "task_index"]) for path in parquet_files]
    )
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    episode_indices = np.asarray(table["episode_index"])
    task_indices = np.asarray(table["task_index"])
    episode_rows = {
        int(episode): np.flatnonzero(episode_indices == episode).tolist()
        for episode in np.unique(episode_indices)
    }
    episode_lengths = {episode: len(rows) for episode, rows in episode_rows.items()}
    task_table = pq.read_table(Path(args.root) / "meta/tasks.parquet").to_pydict()
    task_lookup = dict(zip(task_table["task_index"], task_table["task"]))
    languages = Counter(task_lookup[int(index)] for index in task_indices)
    tasks = Counter()
    for rows in episode_rows.values():
        task = task_lookup[int(task_indices[rows[0]])]
        object_name, bin_name = task.removeprefix("Pick up the ").removesuffix(".").split(" and place it in the ")
        tasks[(object_name, bin_name)] += 1
    summary = {
        "number_of_episodes": meta.total_episodes,
        "number_of_frames": meta.total_frames,
        "fps": meta.fps,
        "image_features": meta.camera_keys,
        "state_shape": meta.features["observation.state"]["shape"],
        "action_shape": meta.features["action"]["shape"],
        "task_count": len(meta.tasks),
        "episode_length": {
            "min": min(episode_lengths.values()),
            "max": max(episode_lengths.values()),
            "mean": float(np.mean(list(episode_lengths.values()))),
        },
        "action_stats": {
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
            "mean": action.mean(axis=0).tolist(),
            "std": action.std(axis=0).tolist(),
        },
        "episodes_per_combination": {f"{obj} -> {bin_name}": count for (obj, bin_name), count in sorted(tasks.items())},
        "language_frame_distribution": dict(languages),
    }
    print("DATASET_SUMMARY=" + json.dumps(summary, indent=2))

    episode = args.episode if args.episode is not None else random.Random(args.seed).choice(sorted(episode_rows))
    rows = episode_rows[episode]
    video_path = Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    first = image_uint8(ds[rows[0]]["observation.images.front"])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video_path}")
    for local_idx, row_idx in enumerate(rows):
        row = ds[row_idx]
        frame = image_uint8(row["observation.images.front"])
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if local_idx % args.print_every == 0:
            a = as_numpy(row["action"])
            print(
                f"ACTION episode={episode} frame={local_idx:03d} "
                f"eef={np.round(a[:3], 4).tolist()} rotation={np.round(a[3:6], 4).tolist()} gripper={a[6]:+.1f}"
            )
    writer.release()
    print(f"EXPORTED_EPISODE episode={episode} frames={len(rows)} video={video_path.resolve()}")


if __name__ == "__main__":
    main()
