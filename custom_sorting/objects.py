"""Dependency-free primitive objects for the sorting scene."""

from __future__ import annotations

import numpy as np
from robosuite.models.objects import BoxObject, CompositeObject, CylinderObject


OBJECT_COLORS = {  # (R, G, B, A)
    "red_cube": (0.90, 0.08, 0.08, 1.0),
    "green_cube": (0.08, 0.75, 0.12, 1.0),
    "cylinder": (0.95, 0.75, 0.08, 1.0),
    "small_box": (0.55, 0.15, 0.80, 1.0),
}


def make_sorting_objects(rng: np.random.Generator) -> dict[str, object]:
    """Create four free-joint, graspable MuJoCo primitives."""
    common = dict(density=350.0, friction=(1.0, 0.005, 0.0001), rng=rng)  # (（mass）, friction (sliding, torsional, rolling))
    return {
        "red_cube": BoxObject(name="red_cube", size=(0.022, 0.022, 0.022), rgba=OBJECT_COLORS["red_cube"], **common),  #MuJoCo box 的 size 通常表示半尺寸。
        "green_cube": BoxObject(
            name="green_cube", size=(0.021, 0.021, 0.026), rgba=OBJECT_COLORS["green_cube"], **common
        ),
        "cylinder": CylinderObject(
            name="cylinder", size=(0.020, 0.035), rgba=OBJECT_COLORS["cylinder"], **common
        ),
        "small_box": BoxObject(
            name="small_box", size=(0.030, 0.018, 0.018), rgba=OBJECT_COLORS["small_box"], **common
        ),
    }


def make_static_bin(name: str, rgba: tuple[float, float, float, float]) -> CompositeObject:
    """Create a fixed, open-top 14 x 14 cm bin from five colored boxes."""
    half_x, half_y, height, wall = 0.07, 0.07, 0.045, 0.006
    geom_types = ["box"] * 5
    geom_sizes = [
        (half_x, half_y, wall / 2),
        (wall / 2, half_y, height / 2),
        (wall / 2, half_y, height / 2),
        (half_x, wall / 2, height / 2),
        (half_x, wall / 2, height / 2),
    ]
    geom_locations = [
        (0, 0, -height / 2 + wall / 2),
        (-half_x + wall / 2, 0, 0),
        (half_x - wall / 2, 0, 0),
        (0, -half_y + wall / 2, 0),
        (0, half_y - wall / 2, 0),
    ]
    return CompositeObject(
        name=name,
        total_size=(half_x, half_y, height / 2),
        geom_types=geom_types,
        geom_sizes=geom_sizes,
        geom_locations=geom_locations,
        geom_names=("base", "left", "right", "front", "back"),
        geom_rgbas=[rgba] * 5,
        locations_relative_to_center=True,
        joints=None,
        density=1000.0,
    )
