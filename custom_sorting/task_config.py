"""Single source of truth for the custom sorting task configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SortingConfig:
    robot: str = "Panda"
    control_freq: int = 20
    horizon: int = 300
    image_size: int = 256
    camera_names: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    table_full_size: tuple[float, float, float] = (0.9, 0.8, 0.05)
    table_offset: tuple[float, float, float] = (0.0, 0.0, 0.8)
    object_names: tuple[str, ...] = ("red_cube", "green_cube", "cylinder", "small_box")
    bin_names: tuple[str, ...] = ("red_bin", "blue_bin")
    bin_positions: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "red_bin": (0.18, -0.24, 0.825),
            "blue_bin": (0.18, 0.24, 0.825),
        }
    )
    spawn_x_range: tuple[float, float] = (-0.20, 0.02)
    spawn_y_range: tuple[float, float] = (-0.20, 0.20)
    min_object_separation: float = 0.095
    action_position_scale: float = 0.05
    action_rotation_scale: float = 0.5
    grasp_lift_height: float = 0.06
    expert_position_gain: float = 12.0
    expert_max_position_command: float = 0.75
    expert_reach_tolerance: float = 0.012
    expert_approach_height: float = 0.12
    expert_lift_height: float = 0.16
    expert_bin_approach_height: float = 0.18
    expert_bin_release_height: float = 0.055
    expert_gripper_settle_steps: int = 15

    @property
    def task_combinations(self) -> tuple[tuple[str, str], ...]:
        return tuple((obj, bin_name) for obj in self.object_names for bin_name in self.bin_names)


def load_config(path: str | Path | None = None) -> SortingConfig:
    """Load YAML overrides while retaining typed defaults."""
    if path is None:
        return SortingConfig()
    with Path(path).open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}
    for key in ("camera_names", "table_full_size", "table_offset", "object_names", "bin_names"):
        if key in raw:
            raw[key] = tuple(raw[key])
    for key in ("spawn_x_range", "spawn_y_range"):
        if key in raw:
            raw[key] = tuple(raw[key])
    if "bin_positions" in raw:
        raw["bin_positions"] = {k: tuple(v) for k, v in raw["bin_positions"].items()}
    unknown = set(raw) - set(SortingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown sorting config keys: {sorted(unknown)}")
    return SortingConfig(**raw)
