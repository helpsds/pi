"""Ground-truth scripted expert used only to generate sorting demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sorting_env import SortingEnv


OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = 1.0


@dataclass
class ExpertStep:
    action: np.ndarray
    phase: str
    position_error: float


class SortingExpert:
    """Interpolating Cartesian state machine; it never teleports the robot or object."""

    PHASES = (
        "open",
        "approach_object",
        "descend_to_grasp",
        "close",
        "lift",
        "approach_bin",
        "descend_to_release",
        "open_release",
        "retreat",
        "done",
    )

    def __init__(self, env: SortingEnv):
        self.env = env
        self.cfg = env.config
        self.phase = "open"
        self.phase_steps = 0
        self.initial_object_pos = self.object_pos.copy()
        self.grasp_target = self.initial_object_pos + np.array((0.0, 0.0, 0.012))
        self.approach_target = self.initial_object_pos + np.array((0.0, 0.0, self.cfg.expert_approach_height))
        self.lift_target = self.initial_object_pos + np.array((0.0, 0.0, self.cfg.expert_lift_height))
        bin_center = np.asarray(self.cfg.bin_positions[env.target_bin], dtype=np.float64)
        self.bin_approach_target = bin_center + np.array((0.0, 0.0, self.cfg.expert_bin_approach_height))
        self.bin_release_target = bin_center + np.array((0.0, 0.0, self.cfg.expert_bin_release_height))
        self.retreat_target = bin_center + np.array((0.0, 0.0, self.cfg.expert_bin_approach_height))

    @property
    def object_pos(self) -> np.ndarray:
        body_id = self.env.object_body_ids[self.env.target_object]
        return np.asarray(self.env.sim.data.body_xpos[body_id], dtype=np.float64)

    def _move(self, target: np.ndarray, gripper: float) -> ExpertStep:
        current = np.asarray(self.env._get_observations(force_update=True)["robot0_eef_pos"], dtype=np.float64)
        error = np.asarray(target) - current
        command = np.clip(
            self.cfg.expert_position_gain * error,
            -self.cfg.expert_max_position_command,
            self.cfg.expert_max_position_command,
        )
        action = np.zeros(7, dtype=np.float32)
        action[:3] = command
        action[6] = gripper
        return ExpertStep(action=action, phase=self.phase, position_error=float(np.linalg.norm(error)))

    def _hold(self, gripper: float) -> ExpertStep:
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper
        return ExpertStep(action=action, phase=self.phase, position_error=0.0)

    def _advance_if_reached(self, step: ExpertStep, next_phase: str, tolerance: float | None = None):
        if step.position_error < (tolerance or self.cfg.expert_reach_tolerance):
            self.phase = next_phase
            self.phase_steps = 0

    def act(self) -> ExpertStep:
        """Return one finite 7D action and advance the state machine when appropriate."""
        self.phase_steps += 1
        if self.phase == "open":
            step = self._hold(OPEN_GRIPPER)
            if self.phase_steps >= self.cfg.expert_gripper_settle_steps:
                self.phase, self.phase_steps = "approach_object", 0
        elif self.phase == "approach_object":
            step = self._move(self.approach_target, OPEN_GRIPPER)
            self._advance_if_reached(step, "descend_to_grasp")
        elif self.phase == "descend_to_grasp":
            step = self._move(self.grasp_target, OPEN_GRIPPER)
            self._advance_if_reached(step, "close", tolerance=0.008)
        elif self.phase == "close":
            step = self._move(self.grasp_target, CLOSE_GRIPPER)
            if self.phase_steps >= self.cfg.expert_gripper_settle_steps:
                self.phase, self.phase_steps = "lift", 0
        elif self.phase == "lift":
            step = self._move(self.lift_target, CLOSE_GRIPPER)
            self._advance_if_reached(step, "approach_bin")
        elif self.phase == "approach_bin":
            step = self._move(self.bin_approach_target, CLOSE_GRIPPER)
            self._advance_if_reached(step, "descend_to_release")
        elif self.phase == "descend_to_release":
            step = self._move(self.bin_release_target, CLOSE_GRIPPER)
            self._advance_if_reached(step, "open_release", tolerance=0.010)
        elif self.phase == "open_release":
            step = self._move(self.bin_release_target, OPEN_GRIPPER)
            if self.phase_steps >= self.cfg.expert_gripper_settle_steps:
                self.phase, self.phase_steps = "retreat", 0
        elif self.phase == "retreat":
            step = self._move(self.retreat_target, OPEN_GRIPPER)
            if step.position_error < self.cfg.expert_reach_tolerance:
                self.phase = "done"
        elif self.phase == "done":
            step = self._hold(OPEN_GRIPPER)
        else:
            raise RuntimeError(f"Unknown expert phase: {self.phase}")

        if step.action.shape != (7,) or not np.isfinite(step.action).all():
            raise RuntimeError(f"Invalid expert action: {step.action}")
        return step
