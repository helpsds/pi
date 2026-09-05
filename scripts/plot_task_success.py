#!/usr/bin/env python3
"""Plot per-task Base vs fine-tuned success rates from the published CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("results/task_success.csv"))
    p.add_argument("--output", type=Path, default=Path("assets/figures/task_success.png"))
    args = p.parse_args()
    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    tasks = [row["task"] for row in rows]
    count = np.asarray([int(row["episodes_per_policy"]) for row in rows])
    base = np.asarray([int(row["base_successes"]) for row in rows]) / count * 100
    ft = np.asarray([int(row["ft_successes"]) for row in rows]) / count * 100

    y = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.barh(y + 0.18, base, height=0.34, label="π0.5 Base", color="#9ca3af")
    ax.barh(y - 0.18, ft, height=0.34, label="π0.5 FT 30k", color="#147d64")
    ax.set(yticks=y, yticklabels=tasks, xlabel="Task success (%)", xlim=(0, 100))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
