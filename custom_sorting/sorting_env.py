"""Fixed-base Panda language-conditioned object sorting environment.

Action semantics are identical for data collection, training and evaluation::

    EEF: 末端执行器
    action[0:3]  normalized Cartesian EEF delta (x, y, z)
    action[3:6]  normalized axis-angle EEF rotation delta (rx, ry, rz)
    action[6]    gripper command (-1 open, +1 close)

The Panda OSC_POSE controller maps [-1, 1] to +/-0.05 m translation and
+/-0.5 rad rotation per control command (configured by robosuite's
``default_panda.json``). No mobile base or control-mode channel exists.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat

from .objects import make_sorting_objects, make_static_bin
from .task_config import SortingConfig


BIN_COLORS = {
    "red_bin": (0.85, 0.05, 0.05, 1.0),
    "blue_bin": (0.05, 0.15, 0.90, 1.0),
}


class SortingEnv(ManipulationEnv):
    """Four-object, two-bin sorting task with an explicitly selected task."""

    def __init__(
        self,
        config: SortingConfig | None = None,
        target_object: str = "red_cube",
        target_bin: str = "blue_bin",
        seed: int | None = None,
        **kwargs,
    ):
        self.config = config or SortingConfig()
        if target_object not in self.config.object_names:
            raise ValueError(f"Unknown target object {target_object!r}; choose from {self.config.object_names}")
        if target_bin not in self.config.bin_names:
            raise ValueError(f"Unknown target bin {target_bin!r}; choose from {self.config.bin_names}")
        self.target_object = target_object
        self.target_bin = target_bin
        self.instruction = f"Pick up the {target_object.replace('_', ' ')} and place it in the {target_bin.replace('_', ' ')}."
        self._episode_seed = seed
        self._grasp_success = False
        self._place_success = False
        self.use_object_obs = True

        defaults = dict(
            robots=self.config.robot,
            gripper_types="default",
            base_types="default",
            initialization_noise=None,
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            control_freq=self.config.control_freq,
            horizon=self.config.horizon,
            ignore_done=False,
            hard_reset=False,
            camera_names=list(self.config.camera_names),
            camera_heights=self.config.image_size,
            camera_widths=self.config.image_size,
            camera_depths=False,
            seed=seed,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)
        if self.action_dim != 7:
            raise RuntimeError(f"SortingEnv requires OSC_POSE 7D action, got action_dim={self.action_dim}")

    @property
    def task_id(self) -> tuple[str, str]:
        return self.target_object, self.target_bin

    def _load_model(self):
        super()._load_model()
        cfg = self.config
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](cfg.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=cfg.table_full_size,
            table_friction=(1.0, 0.005, 0.0001),
            table_offset=cfg.table_offset,
        )
        arena.set_origin((0, 0, 0))
        # A slightly elevated front view keeps both fixed bins in frame.
        arena.set_camera(
            camera_name="agentview",
            #六自由度
            pos=(0.65, 0.0, 1.35), #（xyz）
            quat=(0.653, 0.271, 0.271, 0.653),#四元数
        )

        self.sorting_objects = make_sorting_objects(self.rng)
        self.bins = {
            name: make_static_bin(name, BIN_COLORS[name]) for name in cfg.bin_names
        }
        for name, bin_obj in self.bins.items():
            bin_obj.get_obj().set("pos", array_to_string(cfg.bin_positions[name]))

        all_objects = list(self.sorting_objects.values()) + list(self.bins.values())
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.object_body_ids = {
            name: self.sim.model.body_name2id(obj.root_body)
            for name, obj in self.sorting_objects.items()
        }
        self.bin_body_ids = {
            name: self.sim.model.body_name2id(obj.root_body) for name, obj in self.bins.items()
        }

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        for object_name in self.config.object_names:
            body_id = self.object_body_ids[object_name]

            @sensor(modality="object")
            def object_pos(_obs_cache, body_id=body_id):
                return np.array(self.sim.data.body_xpos[body_id])

            @sensor(modality="object")
            def object_quat(_obs_cache, body_id=body_id):
                return convert_quat(np.array(self.sim.data.body_xquat[body_id]), to="xyzw")

            object_pos.__name__ = f"{object_name}_pos"
            object_quat.__name__ = f"{object_name}_quat"
            for fn in (object_pos, object_quat):
                observables[fn.__name__] = Observable(
                    name=fn.__name__, sensor=fn, sampling_rate=self.control_freq
                )
        return observables

    def _sample_object_xy(self) -> dict[str, np.ndarray]:
        cfg = self.config
        sampled: dict[str, np.ndarray] = {}
        for name in cfg.object_names:
            for _ in range(1000):
                xy = np.array(
                    [
                        self.rng.uniform(*cfg.spawn_x_range),
                        self.rng.uniform(*cfg.spawn_y_range),
                    ],
                    dtype=np.float64,
                )
                if all(np.linalg.norm(xy - previous) >= cfg.min_object_separation for previous in sampled.values()):
                    sampled[name] = xy
                    break
            else:
                raise RuntimeError("Unable to sample non-overlapping reachable object positions")
        return sampled

    def _reset_internal(self):
        super()._reset_internal()
        self._grasp_success = False
        self._place_success = False
        for name, xy in self._sample_object_xy().items():
            obj = self.sorting_objects[name]
            z = self.config.table_offset[2] - obj.bottom_offset[2] + 0.002
            yaw = self.rng.uniform(-np.pi, np.pi)
            quat = np.array((np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
            self.sim.data.set_joint_qpos(obj.joints[0], np.r_[xy, z, quat])

    def reset(self, seed: int | None = None):
        """Reset with deterministic object placements for an explicit seed."""
        if seed is not None:
            self._episode_seed = int(seed)
            self.rng = np.random.default_rng(self._episode_seed)
            for obj in getattr(self, "sorting_objects", {}).values():
                obj.rng = self.rng
        obs = super().reset()
        return self._augment_observation(obs)

    def _augment_observation(self, obs: OrderedDict) -> OrderedDict:
        obs["language_instruction"] = self.instruction
        obs["sorting_state"] = self.get_proprioceptive_state(obs)
        return obs

    def get_proprioceptive_state(self, obs=None) -> np.ndarray:
        """9D policy state: EEF xyz, EEF xyzw quaternion, two gripper joints."""
        if obs is None:
            obs = self._get_observations(force_update=True)
        return np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    def _target_inside_bin(self) -> bool:
        pos = np.asarray(self.sim.data.body_xpos[self.object_body_ids[self.target_object]])
        center = np.asarray(self.config.bin_positions[self.target_bin])
        return bool(
            abs(pos[0] - center[0]) < 0.058
            and abs(pos[1] - center[1]) < 0.058
            and center[2] - 0.01 < pos[2] < center[2] + 0.10
        )

    def success_metrics(self) -> dict[str, bool]:
        obj = self.sorting_objects[self.target_object]
        is_grasped = bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=obj))
        obj_z = float(self.sim.data.body_xpos[self.object_body_ids[self.target_object]][2])
        lifted = obj_z > self.config.table_offset[2] + self.config.grasp_lift_height
        self._grasp_success = self._grasp_success or (is_grasped and lifted)
        self._place_success = self._place_success or self._target_inside_bin()
        task_success = self._place_success and not is_grasped
        return {
            "grasp_success": bool(self._grasp_success),
            "place_success": bool(self._place_success),
            "task_success": bool(task_success),
        }

    def _check_success(self):
        return self.success_metrics()["task_success"]

    def reward(self, action=None):
        return float(self._check_success())

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected 7D [dpos(3), drot(3), gripper(1)], got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError(f"Action contains non-finite values: {action}")
        obs, reward, done, info = super().step(action)
        metrics = self.success_metrics()
        info.update(metrics)
        if metrics["task_success"]:
            done = True
        return self._augment_observation(obs), reward, done, info
