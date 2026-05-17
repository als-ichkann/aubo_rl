from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aubo_kinematics import AuboKinematics, JOINT_NAMES


ACTUATOR_NAMES = (
    "shoulder_pan_act",
    "shoulder_lift_act",
    "elbow_act",
    "wrist_1_act",
    "wrist_2_act",
    "wrist_3_act",
)


def _as_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must have size {size}, got {vector.size}")
    return vector


def _estimate_duration(delta_q: np.ndarray, max_vel: np.ndarray, max_acc: np.ndarray) -> float:
    """Estimate a conservative duration for a smooth fifth-order joint move."""
    abs_delta = np.abs(delta_q)
    vel_time = abs_delta / np.maximum(max_vel, 1e-6)
    acc_time = np.sqrt(abs_delta / np.maximum(max_acc, 1e-6))
    return float(max(0.005, np.max(np.maximum(vel_time, acc_time))))


@dataclass
class SCurveJointTrajectory:
    dof: int = 6
    max_vel: np.ndarray = field(default_factory=lambda: np.full(6, 1.2))
    max_acc: np.ndarray = field(default_factory=lambda: np.full(6, 3.0))
    min_duration: float = 0.015

    q_start: np.ndarray = field(init=False)
    q_goal: np.ndarray = field(init=False)
    start_time: float = field(default=0.0, init=False)
    duration: float = field(default=0.08, init=False)
    active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.max_vel = _as_vector(self.max_vel, self.dof, "max_vel")
        self.max_acc = _as_vector(self.max_acc, self.dof, "max_acc")
        self.q_start = np.zeros(self.dof, dtype=float)
        self.q_goal = np.zeros(self.dof, dtype=float)

    def reset(self, q_start: np.ndarray, q_goal: np.ndarray, now: float) -> None:
        self.q_start = _as_vector(q_start, self.dof, "q_start")
        self.q_goal = _as_vector(q_goal, self.dof, "q_goal")
        self.start_time = float(now)
        estimated = _estimate_duration(self.q_goal - self.q_start, self.max_vel, self.max_acc)
        self.duration = max(self.min_duration, estimated)
        self.active = True

    def sample(self, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        if not self.active:
            zeros = np.zeros(self.dof, dtype=float)
            return self.q_goal.copy(), zeros, zeros, True

        elapsed = max(0.0, float(now) - self.start_time)
        u = np.clip(elapsed / self.duration, 0.0, 1.0)

        # Fifth-order S-curve: zero velocity and acceleration at both ends.
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        ds_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        d2s_du2 = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

        delta = self.q_goal - self.q_start
        q_ref = self.q_start + s * delta
        qd_ref = (ds_du / self.duration) * delta
        qdd_ref = (d2s_du2 / (self.duration**2)) * delta
        done = bool(u >= 1.0)
        if done:
            self.active = False
            q_ref = self.q_goal.copy()
            qd_ref = np.zeros(self.dof, dtype=float)
            qdd_ref = np.zeros(self.dof, dtype=float)
        return q_ref, qd_ref, qdd_ref, done


@dataclass
class InverseDynamicsPDController:
    dof: int = 6
    kp: np.ndarray = field(default_factory=lambda: np.full(6, 80.0))
    kd: np.ndarray = field(default_factory=lambda: np.full(6, 18.0))
    torque_limit: np.ndarray = field(default_factory=lambda: np.array([180.0, 260.0, 220.0, 90.0, 70.0, 35.0]))

    def __post_init__(self) -> None:
        self.kp = _as_vector(self.kp, self.dof, "kp")
        self.kd = _as_vector(self.kd, self.dof, "kd")
        self.torque_limit = _as_vector(self.torque_limit, self.dof, "torque_limit")

    def reset(self) -> None:
        pass

    def compute(
        self,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        qdd_ref: np.ndarray,
        q_current: np.ndarray,
        qvel_current: np.ndarray,
        mass_matrix: np.ndarray,
        bias_force: np.ndarray,
    ) -> np.ndarray:
        q_ref = _as_vector(q_ref, self.dof, "q_ref")
        qd_ref = _as_vector(qd_ref, self.dof, "qd_ref")
        qdd_ref = _as_vector(qdd_ref, self.dof, "qdd_ref")
        q_current = _as_vector(q_current, self.dof, "q_current")
        qvel_current = _as_vector(qvel_current, self.dof, "qvel_current")
        mass_matrix = np.asarray(mass_matrix, dtype=float).reshape(self.dof, self.dof)
        bias_force = _as_vector(bias_force, self.dof, "bias_force")

        error = q_ref - q_current
        d_error = qd_ref - qvel_current
        desired_acc = qdd_ref + self.kp * error + self.kd * d_error
        tau = mass_matrix @ desired_acc + bias_force
        return np.clip(tau, -self.torque_limit, self.torque_limit)


JointPIDServo = InverseDynamicsPDController


@dataclass
class CascadeControllerConfig:
    motion_rate: float = 80.0
    servo_rate: float = 500.0
    ik_max_joint_step: float = 1.2
    target_change_tolerance: float = 1e-4
    max_vel: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0, 5.0, 5.0, 6.0]))
    max_acc: np.ndarray = field(default_factory=lambda: np.array([18.0, 18.0, 18.0, 24.0, 24.0, 30.0]))
    min_trajectory_duration: float = 0.015
    kp: np.ndarray = field(default_factory=lambda: np.array([180.0, 180.0, 180.0, 140.0, 120.0, 80.0]))
    kd: np.ndarray = field(default_factory=lambda: np.array([34.0, 34.0, 34.0, 26.0, 22.0, 16.0]))
    torque_limit: np.ndarray = field(default_factory=lambda: np.array([260.0, 360.0, 320.0, 140.0, 110.0, 70.0]))


@dataclass(frozen=True)
class CascadeControllerState:
    q_ref: np.ndarray
    qd_ref: np.ndarray
    qdd_ref: np.ndarray
    q_goal: np.ndarray
    ctrl: np.ndarray
    ik_success: bool
    trajectory_done: bool
    message: str = ""


class AuboCascadeController:
    def __init__(
        self,
        kinematics: AuboKinematics,
        config: CascadeControllerConfig | None = None,
    ) -> None:
        self.kinematics = kinematics
        self.config = config or CascadeControllerConfig()
        self.motion_dt = 1.0 / max(self.config.motion_rate, 1e-6)
        self.servo_dt = 1.0 / max(self.config.servo_rate, 1e-6)

        self.trajectory = SCurveJointTrajectory(
            dof=6,
            max_vel=self.config.max_vel,
            max_acc=self.config.max_acc,
            min_duration=self.config.min_trajectory_duration,
        )
        self.servo = InverseDynamicsPDController(
            dof=6,
            kp=self.config.kp,
            kd=self.config.kd,
            torque_limit=self.config.torque_limit,
        )

        self.target_pose: np.ndarray | None = None
        self.last_solved_target: np.ndarray | None = None
        self.last_motion_update_time = -np.inf
        self.last_servo_update_time = -np.inf
        self.q_goal = np.zeros(6, dtype=float)
        self.last_ctrl = np.zeros(6, dtype=float)
        self.qpos_addresses: list[int] | None = None
        self.qvel_addresses: list[int] | None = None
        self.actuator_ids: list[int] | None = None
        self.initialized = False
        self.last_state = CascadeControllerState(
            q_ref=np.zeros(6, dtype=float),
            qd_ref=np.zeros(6, dtype=float),
            qdd_ref=np.zeros(6, dtype=float),
            q_goal=np.zeros(6, dtype=float),
            ctrl=np.zeros(6, dtype=float),
            ik_success=True,
            trajectory_done=True,
        )

    def reset(self, q_current: np.ndarray, now: float = 0.0) -> None:
        q_current = _as_vector(q_current, 6, "q_current")
        self.servo.reset()
        self.trajectory.reset(q_current, q_current, now)
        self.q_goal = q_current.copy()
        self.last_ctrl[:] = 0.0
        self.last_solved_target = None
        self.last_motion_update_time = -np.inf
        self.last_servo_update_time = -np.inf
        self.initialized = True

    def set_target_pose(self, target_pose: np.ndarray) -> None:
        self.target_pose = _as_vector(target_pose, 6, "target_pose")

    def bind_mujoco_model(self, model) -> None:
        self.qpos_addresses = get_mujoco_joint_qpos_addresses(model)
        self.qvel_addresses = get_mujoco_joint_qvel_addresses(model)
        self.actuator_ids = get_mujoco_actuator_ids(model)

    def update(
        self,
        q_current: np.ndarray,
        qvel_current: np.ndarray,
        now: float,
        mass_matrix: np.ndarray | None = None,
        bias_force: np.ndarray | None = None,
    ) -> CascadeControllerState:
        q_current = _as_vector(q_current, 6, "q_current")
        qvel_current = _as_vector(qvel_current, 6, "qvel_current")
        now = float(now)

        ik_success = True
        message = ""

        if self.target_pose is not None and now - self.last_motion_update_time >= self.motion_dt:
            target_changed = (
                self.last_solved_target is None
                or np.linalg.norm(self.target_pose - self.last_solved_target) > self.config.target_change_tolerance
            )
            if target_changed or not self.trajectory.active:
                seed_q = self.q_goal if self.trajectory.active else q_current
                ik_result = self.kinematics.solve_ik(
                    self.target_pose,
                    seed_q,
                    max_joint_step=self.config.ik_max_joint_step,
                )
                ik_success = ik_result.success
                message = ik_result.message
                if ik_result.success:
                    q_ref_now, _, _, _ = self.trajectory.sample(now)
                    self.q_goal = ik_result.q.copy()
                    self.trajectory.reset(q_ref_now, self.q_goal, now)
                    self.last_solved_target = self.target_pose.copy()
                    self.servo.reset()
            self.last_motion_update_time = now

        q_ref, qd_ref, qdd_ref, trajectory_done = self.trajectory.sample(now)
        if now - self.last_servo_update_time >= self.servo_dt:
            if mass_matrix is None:
                mass_matrix = np.eye(6)
            if bias_force is None:
                bias_force = np.zeros(6, dtype=float)
            self.last_ctrl = self.servo.compute(
                q_ref,
                qd_ref,
                qdd_ref,
                q_current,
                qvel_current,
                mass_matrix,
                bias_force,
            )
            self.last_servo_update_time = now

        self.last_state = CascadeControllerState(
            q_ref=q_ref.copy(),
            qd_ref=qd_ref.copy(),
            qdd_ref=qdd_ref.copy(),
            q_goal=self.q_goal.copy(),
            ctrl=self.last_ctrl.copy(),
            ik_success=ik_success,
            trajectory_done=trajectory_done,
            message=message,
        )
        return self.last_state

    def update_mujoco(self, model, data, now: float) -> CascadeControllerState:
        import mujoco

        if self.qpos_addresses is None or self.qvel_addresses is None or self.actuator_ids is None:
            self.bind_mujoco_model(model)

        if not self.initialized:
            self.reset(read_mujoco_q(data, self.qpos_addresses), now)

        mujoco.mj_forward(model, data)
        q_current = read_mujoco_q(data, self.qpos_addresses)
        qvel_current = read_mujoco_qvel(data, self.qvel_addresses)

        full_mass = np.zeros((model.nv, model.nv), dtype=float)
        mujoco.mj_fullM(model, full_mass, data.qM)
        mass_matrix = full_mass[np.ix_(self.qvel_addresses, self.qvel_addresses)]
        bias_force = np.asarray(data.qfrc_bias, dtype=float)[self.qvel_addresses]

        state = self.update(q_current, qvel_current, now, mass_matrix, bias_force)
        self.write_ctrl(model, data, self.actuator_ids, state.ctrl)
        return state

    def write_ctrl(self, model, data, actuator_ids: list[int], ctrl: np.ndarray) -> None:
        ctrl = _as_vector(ctrl, 6, "ctrl")
        for idx, actuator_id in enumerate(actuator_ids):
            ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(ctrl[idx], ctrl_min, ctrl_max)


def create_default_controller(
    urdf_path: str | Path,
    end_frame: str = "tool0",
    config: CascadeControllerConfig | None = None,
) -> AuboCascadeController:
    return AuboCascadeController(AuboKinematics(urdf_path, end_frame=end_frame), config=config)


def get_mujoco_joint_qpos_addresses(model, joint_names: tuple[str, ...] = JOINT_NAMES) -> list[int]:
    import mujoco

    addresses = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def get_mujoco_joint_qvel_addresses(model, joint_names: tuple[str, ...] = JOINT_NAMES) -> list[int]:
    import mujoco

    addresses = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        addresses.append(int(model.jnt_dofadr[joint_id]))
    return addresses


def get_mujoco_joint_addresses(model, joint_names: tuple[str, ...] = JOINT_NAMES) -> list[int]:
    return get_mujoco_joint_qpos_addresses(model, joint_names)


def get_mujoco_actuator_ids(model, actuator_names: tuple[str, ...] = ACTUATOR_NAMES) -> list[int]:
    import mujoco

    actuator_ids = []
    for actuator_name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {actuator_name}")
        actuator_ids.append(actuator_id)
    return actuator_ids


def read_mujoco_q(data, qpos_addresses: list[int]) -> np.ndarray:
    return np.array([data.qpos[address] for address in qpos_addresses], dtype=float)


def read_mujoco_qvel(data, qvel_addresses: list[int]) -> np.ndarray:
    return np.array([data.qvel[address] for address in qvel_addresses], dtype=float)
