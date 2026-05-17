from __future__ import annotations

import argparse
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from aubo_cascade_controller import AuboCascadeController, CascadeControllerConfig
from aubo_kinematics import AuboKinematics, JOINT_NAMES, Pose6D, rpy_to_matrix


ACTUATOR_NAMES = (
    "shoulder_pan_act",
    "shoulder_lift_act",
    "elbow_act",
    "wrist_1_act",
    "wrist_2_act",
    "wrist_3_act",
)


KEY_HELP = """
键盘控制:
  W/S: X +/-      A/D: Y +/-      R/F: Z +/-
  I/K: Roll +/-   J/L: Pitch +/-  U/O: Yaw +/-
  +/-: 调整步长    Space: 以当前末端位姿重置目标
  C: 开关末端相机窗口  H: 打印帮助  Q/Esc: 退出

遥控器默认映射:
  Axis0+: X+       Axis1-: Y+       Axis4-: Z+
  Axis6+: Roll+    Axis7-: Pitch+   Axis3+: Yaw+
  按钮0: 开关相机   按钮1: 退出        按钮3: 重置目标
"""


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)


@dataclass
class TeleopState:
    target_pose: np.ndarray
    position_step: float = 0.01
    rotation_step: float = np.deg2rad(3.0)
    version: int = 0
    show_camera: bool = False
    running: bool = True
    reset_requested: bool = False

    def apply_key(self, keycode: int) -> None:
        if keycode in (27, 256, ord("Q")):
            self.running = False
            return

        if keycode == 32:
            self.reset_requested = True
            return

        try:
            key = chr(keycode).lower()
        except ValueError:
            return

        delta = np.zeros(6, dtype=float)
        if key == "q":
            self.running = False
            return
        elif key == "w":
            delta[0] += self.position_step
        elif key == "s":
            delta[0] -= self.position_step
        elif key == "a":
            delta[1] += self.position_step
        elif key == "d":
            delta[1] -= self.position_step
        elif key == "r":
            delta[2] += self.position_step
        elif key == "f":
            delta[2] -= self.position_step
        elif key == "i":
            delta[3] += self.rotation_step
        elif key == "k":
            delta[3] -= self.rotation_step
        elif key == "j":
            delta[4] += self.rotation_step
        elif key == "l":
            delta[4] -= self.rotation_step
        elif key == "u":
            delta[5] += self.rotation_step
        elif key == "o":
            delta[5] -= self.rotation_step
        elif key in ("+", "="):
            self.position_step = min(self.position_step * 1.25, 0.05)
            self.rotation_step = min(self.rotation_step * 1.25, np.deg2rad(15.0))
            print_step(self)
            return
        elif key in ("-", "_"):
            self.position_step = max(self.position_step / 1.25, 0.001)
            self.rotation_step = max(self.rotation_step / 1.25, np.deg2rad(0.5))
            print_step(self)
            return
        elif key == "c":
            self.show_camera = not self.show_camera
            print(f"末端相机窗口: {'开启' if self.show_camera else '关闭'}")
            return
        elif key == "h":
            print(KEY_HELP)
            return
        else:
            return

        self.target_pose += delta
        self.target_pose[3:6] = wrap_to_pi(self.target_pose[3:6])
        self.version += 1

    def apply_delta(self, delta: np.ndarray) -> None:
        if np.allclose(delta, 0.0):
            return
        self.target_pose += delta
        self.target_pose[3:6] = wrap_to_pi(self.target_pose[3:6])
        self.version += 1


class LinuxJoystick:
    def __init__(
        self,
        device: str,
        axis_deadzone: float = 0.12,
        button_reset: int = 3,
        button_camera: int = 0,
        button_quit: int = 1,
    ) -> None:
        self.device = device
        self.axis_deadzone = axis_deadzone
        self.button_reset = button_reset
        self.button_camera = button_camera
        self.button_quit = button_quit
        self.axes: dict[int, float] = {}
        self.buttons: dict[int, int] = {}
        self._fd: int | None = None

    def open(self) -> bool:
        try:
            self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            print(f"未启用遥控器输入: 无法打开 {self.device}: {exc}")
            return False
        print(f"遥控器输入已启用: {self.device}")
        return True

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def poll(self, state: TeleopState) -> None:
        if self._fd is None:
            return

        while True:
            try:
                event = os.read(self._fd, JS_EVENT_SIZE)
            except BlockingIOError:
                return
            except OSError as exc:
                print(f"遥控器读取失败，已停用: {exc}")
                self.close()
                return

            if len(event) != JS_EVENT_SIZE:
                return

            _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, event)
            event_type &= ~JS_EVENT_INIT

            if event_type == JS_EVENT_AXIS:
                axis_value = float(np.clip(value / 32767.0, -1.0, 1.0))
                self.axes[number] = 0.0 if abs(axis_value) < self.axis_deadzone else axis_value
            elif event_type == JS_EVENT_BUTTON:
                previous = self.buttons.get(number, 0)
                self.buttons[number] = value
                if value == 1 and previous == 0:
                    self._handle_button(number, state)

    def axis(self, number: int, invert: bool = False) -> float:
        value = self.axes.get(number, 0.0)
        return -value if invert else value

    def _handle_button(self, number: int, state: TeleopState) -> None:
        if number == self.button_reset:
            state.reset_requested = True
        elif number == self.button_camera:
            state.show_camera = not state.show_camera
            print(f"末端相机窗口: {'开启' if state.show_camera else '关闭'}")
        elif number == self.button_quit:
            state.running = False


class EndEffectorCamera:
    def __init__(self, model: mujoco.MjModel, camera_name: str, width: int, height: int) -> None:
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData, pose: Pose6D, target_pose: np.ndarray) -> int:
        import cv2

        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            format_status_line("cur", pose.as_vector()),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            format_status_line("tar", target_pose),
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        cv2.imshow("MuJoCo ego_camera", frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        try:
            import cv2

            cv2.destroyWindow("MuJoCo ego_camera")
        except Exception:
            pass


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def print_step(state: TeleopState) -> None:
    print(
        "当前步长: "
        f"平移 {state.position_step:.4f} m, "
        f"旋转 {np.rad2deg(state.rotation_step):.2f} deg"
    )


def format_status_line(prefix: str, pose: np.ndarray) -> str:
    rpy_deg = np.rad2deg(pose[3:6])
    return (
        f"{prefix} xyz=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) "
        f"rpy=({rpy_deg[0]:+.1f},{rpy_deg[1]:+.1f},{rpy_deg[2]:+.1f})"
    )


def pose_error_norm(current_pose: Pose6D, target_pose: np.ndarray) -> tuple[float, float]:
    position_error = float(np.linalg.norm(target_pose[:3] - current_pose.position))
    rotation_error = float(np.linalg.norm(wrap_to_pi(target_pose[3:6] - current_pose.rpy)))
    return position_error, rotation_error


def get_joint_addresses(model: mujoco.MjModel) -> list[int]:
    addresses = []
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def get_actuator_ids(model: mujoco.MjModel) -> list[int]:
    actuator_ids = []
    for actuator_name in ACTUATOR_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {actuator_name}")
        actuator_ids.append(actuator_id)
    return actuator_ids


def read_mujoco_q(data: mujoco.MjData, qpos_addresses: list[int]) -> np.ndarray:
    return np.array([data.qpos[address] for address in qpos_addresses], dtype=float)


def write_velocity_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
    q_current: np.ndarray,
    q_desired: np.ndarray,
    gain: float,
) -> None:
    velocity = gain * (q_desired - q_current)
    for idx, actuator_id in enumerate(actuator_ids):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(velocity[idx], ctrl_min, ctrl_max)


def apply_joystick_motion(
    state: TeleopState,
    joystick: LinuxJoystick,
    dt: float,
    args: argparse.Namespace,
) -> None:
    delta = np.array(
        [
            joystick.axis(args.axis_x, invert=args.invert_x) * args.joystick_xyz_speed * dt,
            joystick.axis(args.axis_y, invert=args.invert_y) * args.joystick_xyz_speed * dt,
            joystick.axis(args.axis_z, invert=args.invert_z) * args.joystick_xyz_speed * dt,
            joystick.axis(args.axis_roll, invert=args.invert_roll) * args.joystick_rpy_speed * dt,
            joystick.axis(args.axis_pitch, invert=args.invert_pitch) * args.joystick_rpy_speed * dt,
            joystick.axis(args.axis_yaw, invert=args.invert_yaw) * args.joystick_rpy_speed * dt,
        ],
        dtype=float,
    )
    state.apply_delta(delta)


def rotation_from_z_axis(direction: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(z_axis)
    if norm < 1e-9:
        return np.eye(3)
    z_axis = z_axis / norm

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, z_axis)) > 0.95:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def add_arrow_geom(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([radius, radius, length], dtype=float),
        np.asarray(position, dtype=float),
        rotation_from_z_axis(direction).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def add_capsule_segment(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=float)
    scene.ngeom += 1


def add_rotation_arc(
    scene: mujoco.MjvScene,
    axis: str,
    center: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    angles = np.linspace(0.0, np.deg2rad(30.0), 8)
    points = []
    for angle in angles:
        if axis == "roll":
            point = center + np.array([0.0, radius * np.cos(angle), radius * np.sin(angle)])
        elif axis == "pitch":
            point = center + np.array([radius * np.sin(angle), 0.0, radius * np.cos(angle)])
        else:
            point = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        points.append(point)

    for start, end in zip(points[:-1], points[1:]):
        add_capsule_segment(scene, start, end, 0.0025, rgba)

    tangent = points[-1] - points[-2]
    add_arrow_geom(scene, points[-1], tangent, 0.05, 0.005, rgba)


def draw_base_reference_markers(scene: mujoco.MjvScene) -> None:
    origin = np.array([0.0, 0.0, 0.06], dtype=float)
    arc_center = np.array([0.0, 0.0, 0.18], dtype=float)
    red = np.array([0.9, 0.1, 0.1, 0.85], dtype=float)
    green = np.array([0.1, 0.8, 0.1, 0.85], dtype=float)
    blue = np.array([0.1, 0.25, 0.95, 0.85], dtype=float)

    add_arrow_geom(scene, origin, np.array([1.0, 0.0, 0.0]), 0.45, 0.009, red)
    add_arrow_geom(scene, origin, np.array([0.0, 1.0, 0.0]), 0.45, 0.009, green)
    add_arrow_geom(scene, origin, np.array([0.0, 0.0, 1.0]), 0.45, 0.009, blue)

    add_rotation_arc(scene, "roll", arc_center, 0.12, red)
    add_rotation_arc(scene, "pitch", arc_center, 0.145, green)
    add_rotation_arc(scene, "yaw", arc_center, 0.17, blue)


def draw_target_marker(scene: mujoco.MjvScene, target_pose: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    position = target_pose[:3].astype(float)
    rotation = rpy_to_matrix(*target_pose[3:6]).reshape(-1)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.006, 0.006, 0.09], dtype=float),
        position,
        rotation,
        np.array([0.1, 0.8, 0.1, 0.65], dtype=float),
    )
    scene.ngeom += 1


def draw_scene_markers(viewer: mujoco.viewer.Handle, target_pose: np.ndarray) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    draw_base_reference_markers(scene)
    draw_target_marker(scene, target_pose)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="AUBO MuJoCo + Pinocchio/CasADi keyboard teleop")
    parser.add_argument("--scene", default=str(root / "scene.xml"), help="MuJoCo scene XML path")
    parser.add_argument(
        "--urdf",
        default=str(root / "models" / "auboi5_mujoco_kinematics.urdf"),
        help="Pinocchio kinematics URDF path",
    )
    parser.add_argument("--end-frame", default="tool0", help="Pinocchio end-effector frame")
    parser.add_argument("--camera", action="store_true", help="Show ego_camera OpenCV window")
    parser.add_argument("--camera-name", default="ego_camera", help="MuJoCo camera name")
    parser.add_argument("--gain", type=float, default=180.0, help="Inverse-dynamics PD position gain")
    parser.add_argument("--damping", type=float, default=34.0, help="Inverse-dynamics PD velocity gain")
    parser.add_argument("--fps", type=float, default=120.0, help="Simulation loop rate")
    parser.add_argument("--ik-rate", type=float, default=80.0, help="IK update rate in Hz")
    parser.add_argument("--ik-step", type=float, default=1.2, help="Max joint change per IK update in rad")
    parser.add_argument("--max-vel", type=float, default=4.0, help="S-curve max joint velocity in rad/s")
    parser.add_argument("--max-acc", type=float, default=18.0, help="S-curve max joint acceleration in rad/s^2")
    parser.add_argument("--min-trajectory-duration", type=float, default=0.015, help="Minimum S-curve segment duration in s")
    parser.add_argument("--sim-steps", type=int, default=4, help="MuJoCo physics steps per control loop")
    parser.add_argument("--joystick", default="/dev/input/js0", help="Linux joystick device path")
    parser.add_argument("--no-joystick", action="store_true", help="Disable joystick input")
    parser.add_argument("--joystick-deadzone", type=float, default=0.06, help="Joystick axis deadzone")
    parser.add_argument("--joystick-xyz-speed", type=float, default=0.25, help="Joystick xyz target speed in m/s")
    parser.add_argument("--joystick-rpy-speed", type=float, default=1.6, help="Joystick rpy target speed in rad/s")
    parser.add_argument("--axis-x", type=int, default=0, help="Joystick axis for world X")
    parser.add_argument("--axis-y", type=int, default=1, help="Joystick axis for world Y")
    parser.add_argument("--axis-z", type=int, default=4, help="Joystick axis for world Z")
    parser.add_argument("--axis-roll", type=int, default=6, help="Joystick axis for roll")
    parser.add_argument("--axis-pitch", type=int, default=7, help="Joystick axis for pitch")
    parser.add_argument("--axis-yaw", type=int, default=3, help="Joystick axis for yaw")
    parser.add_argument("--invert-x", action=argparse.BooleanOptionalAction, default=False, help="Invert X axis")
    parser.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=True, help="Invert Y axis")
    parser.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True, help="Invert Z axis")
    parser.add_argument("--invert-roll", action=argparse.BooleanOptionalAction, default=False, help="Invert roll axis")
    parser.add_argument("--invert-pitch", action=argparse.BooleanOptionalAction, default=True, help="Invert pitch axis")
    parser.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=False, help="Invert yaw axis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    kinematics = AuboKinematics(args.urdf, end_frame=args.end_frame)
    controller = AuboCascadeController(
        kinematics,
        CascadeControllerConfig(
            motion_rate=args.ik_rate,
            servo_rate=args.fps,
            ik_max_joint_step=args.ik_step,
            max_vel=np.full(6, args.max_vel),
            max_acc=np.full(6, args.max_acc),
            min_trajectory_duration=args.min_trajectory_duration,
            kp=np.full(6, args.gain),
            kd=np.full(6, args.damping),
        ),
    )

    qpos_addresses = get_joint_addresses(model)
    mujoco.mj_forward(model, data)

    current_q = read_mujoco_q(data, qpos_addresses)
    current_pose = kinematics.fk(current_q)
    state = TeleopState(target_pose=current_pose.as_vector(), show_camera=args.camera)
    controller.reset(current_q, time.time())
    controller.set_target_pose(state.target_pose)
    camera: EndEffectorCamera | None = None
    last_status_time = 0.0
    last_loop_time = time.time()
    dt = 1.0 / max(args.fps, 1.0)
    joystick = None
    if not args.no_joystick:
        joystick = LinuxJoystick(args.joystick, axis_deadzone=args.joystick_deadzone)
        if not joystick.open():
            joystick = None

    print(KEY_HELP)
    print(format_status_line("cur", current_pose.as_vector()))

    def key_callback(keycode: int) -> None:
        state.apply_key(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and state.running:
            loop_start = time.time()
            loop_dt = np.clip(loop_start - last_loop_time, 1e-4, 0.05)
            last_loop_time = loop_start
            current_q = read_mujoco_q(data, qpos_addresses)
            current_pose = kinematics.fk(current_q)

            if joystick is not None:
                joystick.poll(state)
                apply_joystick_motion(state, joystick, loop_dt, args)

            if state.reset_requested:
                state.target_pose = current_pose.as_vector()
                state.version += 1
                state.reset_requested = False
                controller.reset(current_q, time.time())
                print("目标位姿已重置为当前末端位姿")

            now = time.time()
            controller.set_target_pose(state.target_pose)
            controller_state = controller.update_mujoco(model, data, now)
            reachable_target_pose = kinematics.fk(controller_state.q_goal).as_vector()
            raw_pos_err, raw_rot_err = pose_error_norm(current_pose, state.target_pose)
            reach_pos_err, reach_rot_err = pose_error_norm(current_pose, reachable_target_pose)
            draw_scene_markers(viewer, reachable_target_pose)

            if state.show_camera:
                if camera is None:
                    camera = EndEffectorCamera(model, args.camera_name, 640, 480)
                key = camera.render(data, current_pose, state.target_pose)
                if key not in (255, -1):
                    state.apply_key(key)
            elif camera is not None:
                camera.close()
                camera = None

            if now - last_status_time > 1.0:
                print(format_status_line("cur", current_pose.as_vector()))
                print(format_status_line("cmd", state.target_pose))
                print(format_status_line("ik ", reachable_target_pose))
                print(
                    f"err raw pos={raw_pos_err:.4f} m, rot={np.rad2deg(raw_rot_err):.2f} deg | "
                    f"ik pos={reach_pos_err:.4f} m, rot={np.rad2deg(reach_rot_err):.2f} deg"
                )
                print(f"tau max={np.max(np.abs(controller_state.ctrl)):.2f}, ik={controller_state.ik_success}")
                last_status_time = now

            for _ in range(max(1, args.sim_steps)):
                mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    if camera is not None:
        camera.close()
    if joystick is not None:
        joystick.close()


if __name__ == "__main__":
    main()
