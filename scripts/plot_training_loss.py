#!/usr/bin/env python3
"""Rebuild and plot loss from an initial log plus an optional resumed log."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOSS = re.compile(r"ot_train\.py:769 .*?loss:([0-9.eE+-]+)")


def losses(path: Path) -> list[float]:
    return [float(value) for value in LOSS.findall(path.read_text(errors="replace"))]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--initial-log", type=Path, required=True)
    p.add_argument("--resume-log", type=Path)
    p.add_argument("--resume-step", type=int, default=10000)
    p.add_argument("--log-freq", type=int, default=10)
    p.add_argument("--smooth", type=int, default=51)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()

    first = losses(args.initial_log)
    first = first[: args.resume_step // args.log_freq]
    values = first
    if args.resume_log:
        values += losses(args.resume_log)
    step = np.arange(1, len(values) + 1) * args.log_freq
    raw = np.asarray(values)
    smooth = moving_average(raw, min(args.smooth, max(1, len(raw) // 4 * 2 + 1)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        writer.writerows(zip(step, raw, strict=True))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(step, raw, color="#8fb9df", alpha=0.25, linewidth=0.7, label="logged loss")
    ax.plot(step, smooth, color="#145da0", linewidth=2, label=f"moving average ({args.smooth})")
    ax.axvline(args.resume_step, color="#d1495b", linestyle="--", linewidth=1.2, label="resume checkpoint")
    ax.set(title="π0.5 Sorting fine-tuning", xlabel="Optimizer step", ylabel="Flow-matching loss")
    ax.set_xlim(0, step[-1])
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
