from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class Pose6D:
    position: np.ndarray
    rpy: np.ndarray
    rotation: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate((self.position, self.rpy))


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    cost: float
    message: str = ""


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert a ZYX rotation matrix to roll, pitch, yaw."""
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    sy = -r[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-r[0, 1], r[1, 1])

    return np.array([roll, pitch, yaw], dtype=float)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def rpy_to_matrix_ca(roll: ca.MX, pitch: ca.MX, yaw: ca.MX) -> ca.MX:
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw), ca.sin(yaw)

    return ca.vertcat(
        ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        ca.horzcat(-sp, cp * sr, cp * cr),
    )


class AuboKinematics:
    def __init__(
        self,
        urdf_path: str | Path,
        end_frame: str = "tool0",
        joint_names: tuple[str, ...] = JOINT_NAMES,
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.end_frame = end_frame
        self.joint_names = joint_names

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(end_frame)
        if self.frame_id >= self.model.nframes:
            raise ValueError(f"Frame not found in URDF: {end_frame}")

        actual_joint_names = tuple(self.model.names[1:])
        if actual_joint_names != joint_names:
            raise ValueError(
                "URDF joint order does not match MuJoCo actuator order: "
                f"{actual_joint_names} != {joint_names}"
            )

        self.lower_limits = np.asarray(self.model.lowerPositionLimit, dtype=float)
        self.upper_limits = np.asarray(self.model.upperPositionLimit, dtype=float)
        self.neutral_q = pin.neutral(self.model)

        self._build_casadi_fk()
        self._build_ik_solver()

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower_limits, self.upper_limits)

    def fk(self, q: np.ndarray) -> Pose6D:
        q = self.clip_q(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        rotation = np.array(placement.rotation, dtype=float)
        position = np.array(placement.translation, dtype=float)
        return Pose6D(position=position, rpy=matrix_to_rpy(rotation), rotation=rotation)

    def solve_ik(
        self,
        target_pose: np.ndarray,
        seed_q: np.ndarray,
        max_joint_step: float = 0.35,
    ) -> IKResult:
        target_pose = np.asarray(target_pose, dtype=float).reshape(6)
        seed_q = self.clip_q(seed_q)

        self._opti.set_value(self._target_pose_param, target_pose)
        self._opti.set_value(self._seed_q_param, seed_q)
        self._opti.set_initial(self._q_var, seed_q)

        try:
            solution = self._opti.solve()
            q_raw = np.asarray(solution.value(self._q_var), dtype=float).reshape(self.model.nq)
            cost = float(solution.value(self._cost))
            success = True
            message = "ik solved"
        except RuntimeError as exc:
            try:
                q_raw = np.asarray(self._opti.debug.value(self._q_var), dtype=float).reshape(self.model.nq)
            except RuntimeError:
                q_raw = seed_q
            cost = float("inf")
            success = False
            message = str(exc).splitlines()[0]

        delta = np.clip(q_raw - seed_q, -max_joint_step, max_joint_step)
        q_safe = self.clip_q(seed_q + delta)
        return IKResult(q=q_safe, success=success, cost=cost, message=message)

    def _build_casadi_fk(self) -> None:
        q_symbol = ca.SX.sym("q", self.model.nq)
        casadi_model = cpin.Model(self.model)
        casadi_data = casadi_model.createData()

        cpin.framesForwardKinematics(casadi_model, casadi_data, q_symbol)
        placement = casadi_data.oMf[self.frame_id]

        self._fk_fun = ca.Function(
            "auboi5_fk",
            [q_symbol],
            [placement.translation, placement.rotation],
            ["q"],
            ["position", "rotation"],
        )

    def _build_ik_solver(self) -> None:
        self._opti = ca.Opti()
        self._q_var = self._opti.variable(self.model.nq)
        self._target_pose_param = self._opti.parameter(6)
        self._seed_q_param = self._opti.parameter(self.model.nq)

        position, rotation = self._fk_fun(self._q_var)
        target_position = self._target_pose_param[0:3]
        target_rotation = rpy_to_matrix_ca(
            self._target_pose_param[3],
            self._target_pose_param[4],
            self._target_pose_param[5],
        )

        position_error = position - target_position
        rotation_error = rotation - target_rotation
        seed_error = self._q_var - self._seed_q_param

        self._cost = (
            100.0 * ca.sumsqr(position_error)
            + 10.0 * ca.sumsqr(rotation_error)
            + 1e-3 * ca.sumsqr(seed_error)
        )
        self._opti.minimize(self._cost)

        for idx in range(self.model.nq):
            self._opti.subject_to(
                self._opti.bounded(
                    float(self.lower_limits[idx]),
                    self._q_var[idx],
                    float(self.upper_limits[idx]),
                )
            )

        solver_options = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 80,
                "tol": 1e-5,
                "acceptable_tol": 1e-4,
            },
        }
        self._opti.solver("ipopt", solver_options)
