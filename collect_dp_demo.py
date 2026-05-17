from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import zarr
from numcodecs import Blosc

from aubo_cascade_controller import AuboCascadeController, CascadeControllerConfig
from aubo_kinematics import AuboKinematics
from keyboard_teleop_pinocchio import (
    LinuxJoystick,
    TeleopState,
    apply_joystick_motion,
    draw_scene_markers,
    format_status_line,
    get_joint_addresses,
    pose_error_norm,
    read_mujoco_q,
    wrap_to_pi,
)


def pose_delta(next_pose: np.ndarray, prev_pose: np.ndarray) -> np.ndarray:
    delta = np.asarray(next_pose, dtype=float) - np.asarray(prev_pose, dtype=float)
    delta[3:6] = wrap_to_pi(delta[3:6])
    return delta.astype(np.float32)


class ZarrDemoWriter:
    def __init__(self, path: str | Path, image_shape: tuple[int, int, int], chunk_size: int = 128) -> None:
        self.path = Path(path)
        self.image_shape = image_shape
        self.root = zarr.open_group(str(self.path), mode="w")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

        self.image = self.data.create_dataset(
            "image",
            shape=(0, *image_shape),
            chunks=(chunk_size, *image_shape),
            dtype="u1",
            compressor=compressor,
            overwrite=True,
        )
        self.action = self.data.create_dataset(
            "action",
            shape=(0, 6),
            chunks=(chunk_size, 6),
            dtype="f4",
            compressor=compressor,
            overwrite=True,
        )
        self.eef_pose = self.data.create_dataset(
            "eef_pose",
            shape=(0, 6),
            chunks=(chunk_size, 6),
            dtype="f4",
            compressor=compressor,
            overwrite=True,
        )
        self.episode_ends = self.meta.create_dataset(
            "episode_ends",
            shape=(0,),
            chunks=(max(1, chunk_size // 8),),
            dtype="i8",
            overwrite=True,
        )

    @property
    def steps(self) -> int:
        return int(self.action.shape[0])

    @property
    def episodes(self) -> int:
        return int(self.episode_ends.shape[0])

    def append_step(self, image: np.ndarray, action: np.ndarray, eef_pose: np.ndarray) -> None:
        image = np.asarray(image, dtype=np.uint8).reshape(self.image_shape)
        action = np.asarray(action, dtype=np.float32).reshape(1, 6)
        eef_pose = np.asarray(eef_pose, dtype=np.float32).reshape(1, 6)
        self.image.append(image.reshape(1, *self.image_shape), axis=0)
        self.action.append(action, axis=0)
        self.eef_pose.append(eef_pose, axis=0)

    def finish_episode(self) -> None:
        ends = np.asarray([self.steps], dtype=np.int64)
        self.episode_ends.append(ends, axis=0)

    def truncate(self, step_count: int, episode_count: int) -> None:
        self.image.resize((step_count, *self.image_shape))
        self.action.resize((step_count, 6))
        self.eef_pose.resize((step_count, 6))
        self.episode_ends.resize((episode_count,))


@dataclass
class RecordingState:
    recording: bool = False
    episode_start_step: int = 0
    episode_start_count: int = 0
    prev_target_pose: np.ndarray | None = None

    def start(self, writer: ZarrDemoWriter, target_pose: np.ndarray) -> None:
        if self.recording:
            return
        self.recording = True
        self.episode_start_step = writer.steps
        self.episode_start_count = writer.episodes
        self.prev_target_pose = np.asarray(target_pose, dtype=float).copy()
        print(f"[record] start episode {writer.episodes + 1}")

    def finish(self, writer: ZarrDemoWriter) -> None:
        if not self.recording:
            return
        if writer.steps > self.episode_start_step:
            writer.finish_episode()
            print(f"[record] finish episode {writer.episodes}, total steps={writer.steps}")
        else:
            print("[record] empty episode ignored")
        self.recording = False
        self.prev_target_pose = None

    def discard(self, writer: ZarrDemoWriter) -> None:
        if self.recording:
            writer.truncate(self.episode_start_step, self.episode_start_count)
            self.recording = False
            self.prev_target_pose = None
            print("[record] discarded current episode")


class DemoJoystick(LinuxJoystick):
    def __init__(
        self,
        *args,
        recorder: RecordingState,
        writer: ZarrDemoWriter,
        state: TeleopState,
        button_record: int = 2,
        button_discard: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self.writer = writer
        self.state = state
        self.button_record = button_record
        self.button_discard = button_discard

    def _handle_button(self, number: int, state: TeleopState) -> None:
        if number == self.button_record:
            if self.recorder.recording:
                self.recorder.finish(self.writer)
            else:
                self.recorder.start(self.writer, self.state.target_pose)
            return
        if number == self.button_discard:
            self.recorder.discard(self.writer)
            return
        super()._handle_button(number, state)


def render_camera(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str, image_size: int) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Collect AUBO diffusion policy demonstrations")
    parser.add_argument("--scene", default=str(root / "scene.xml"))
    parser.add_argument("--urdf", default=str(root / "models" / "auboi5_mujoco_kinematics.urdf"))
    parser.add_argument("--output", default=str(root / "expert_demos" / "aubo_ego_rgb_delta_pose.zarr"))
    parser.add_argument("--camera-name", default="ego_camera")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--sim-steps", type=int, default=4)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joystick", default="/dev/input/js0")
    parser.add_argument("--no-joystick", action="store_true")
    parser.add_argument("--joystick-deadzone", type=float, default=0.06)
    parser.add_argument("--joystick-xyz-speed", type=float, default=0.25)
    parser.add_argument("--joystick-rpy-speed", type=float, default=1.6)
    parser.add_argument("--axis-x", type=int, default=0)
    parser.add_argument("--axis-y", type=int, default=1)
    parser.add_argument("--axis-z", type=int, default=4)
    parser.add_argument("--axis-roll", type=int, default=6)
    parser.add_argument("--axis-pitch", type=int, default=7)
    parser.add_argument("--axis-yaw", type=int, default=3)
    parser.add_argument("--invert-x", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--invert-roll", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--invert-pitch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--button-record", type=int, default=2)
    parser.add_argument("--button-discard", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    kinematics = AuboKinematics(args.urdf)
    controller = AuboCascadeController(
        kinematics,
        CascadeControllerConfig(motion_rate=args.sample_rate, servo_rate=args.fps),
    )

    renderer = mujoco.Renderer(model, height=args.image_size, width=args.image_size)
    writer = ZarrDemoWriter(args.output, (args.image_size, args.image_size, 3))
    recorder = RecordingState()
    qpos_addresses = get_joint_addresses(model)
    mujoco.mj_forward(model, data)
    current_q = read_mujoco_q(data, qpos_addresses)
    current_pose = kinematics.fk(current_q)
    state = TeleopState(target_pose=current_pose.as_vector())
    controller.reset(current_q, time.time())
    controller.set_target_pose(state.target_pose)

    joystick: DemoJoystick | None = None
    if not args.no_joystick:
        joystick = DemoJoystick(
            args.joystick,
            axis_deadzone=args.joystick_deadzone,
            recorder=recorder,
            writer=writer,
            state=state,
            button_record=args.button_record,
            button_discard=args.button_discard,
        )
        if not joystick.open():
            joystick = None

    last_loop_time = time.time()
    last_sample_time = 0.0
    dt = 1.0 / max(args.fps, 1.0)
    sample_dt = 1.0 / max(args.sample_rate, 1.0)

    print("[keys] g: start/finish episode, x: discard episode, space: reset target, q: quit")
    print(f"[dataset] writing to {args.output}")

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except ValueError:
            return
        if key == "q":
            state.running = False
        elif key == "g":
            if recorder.recording:
                recorder.finish(writer)
            else:
                recorder.start(writer, state.target_pose)
        elif key == "x":
            recorder.discard(writer)
        else:
            state.apply_key(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and state.running:
            loop_start = time.time()
            loop_dt = np.clip(loop_start - last_loop_time, 1e-4, 0.05)
            last_loop_time = loop_start

            if joystick is not None:
                joystick.poll(state)
                apply_joystick_motion(state, joystick, loop_dt, args)

            current_q = read_mujoco_q(data, qpos_addresses)
            current_pose = kinematics.fk(current_q)
            if state.reset_requested:
                state.target_pose = current_pose.as_vector()
                state.reset_requested = False
                controller.reset(current_q, loop_start)

            controller.set_target_pose(state.target_pose)
            controller_state = controller.update_mujoco(model, data, loop_start)
            reachable_target_pose = kinematics.fk(controller_state.q_goal).as_vector()
            draw_scene_markers(viewer, reachable_target_pose)

            if recorder.recording and loop_start - last_sample_time >= sample_dt:
                image = render_camera(renderer, data, args.camera_name, args.image_size)
                prev_pose = recorder.prev_target_pose
                if prev_pose is None:
                    prev_pose = state.target_pose.copy()
                action = pose_delta(state.target_pose, prev_pose)
                writer.append_step(image, action, current_pose.as_vector())
                recorder.prev_target_pose = state.target_pose.copy()
                last_sample_time = loop_start

            if args.preview:
                image = render_camera(renderer, data, args.camera_name, args.image_size)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                status = "REC" if recorder.recording else "IDLE"
                cv2.putText(frame, status, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("DP demo camera", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    state.running = False
                elif key == ord("g"):
                    if recorder.recording:
                        recorder.finish(writer)
                    else:
                        recorder.start(writer, state.target_pose)
                elif key == ord("x"):
                    recorder.discard(writer)

            for _ in range(max(1, args.sim_steps)):
                mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    if recorder.recording:
        recorder.finish(writer)
    if joystick is not None:
        joystick.close()
    cv2.destroyAllWindows()
    print(f"[dataset] episodes={writer.episodes}, steps={writer.steps}, path={writer.path}")


if __name__ == "__main__":
    main()
