#!/usr/bin/env python3
"""MuJoCo closed-loop simulation: load Diffusion Policy ckpt and track delta pose targets."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import warnings
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DP_ROOT = ROOT / "third_party" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*already registered; second conversion method ignored.*",
)


def _configure_qt_font_env() -> None:
    """OpenCV highgui uses Qt; avoid QFontDatabase spam when cv2/qt/fonts is empty."""
    if os.environ.get("QT_QPA_FONTDIR"):
        return
    for d in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts",
    ):
        if Path(d).is_dir():
            os.environ["QT_QPA_FONTDIR"] = d
            break


def _populate_cv2_qt_fonts() -> None:
    """Qt bundled with pip opencv looks under site-packages/cv2/qt/fonts."""
    try:
        import cv2
    except ImportError:
        return
    qt_fonts = Path(cv2.__file__).resolve().parent / "qt" / "fonts"
    try:
        qt_fonts.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    try:
        if any(qt_fonts.iterdir()):
            return
    except OSError:
        return

    env_font = os.environ.get("QT_QPA_FONTDIR")
    src_dirs = [Path(env_font)] if env_font else []
    src_dirs.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/truetype/liberation"),
        ]
    )
    for src in src_dirs:
        if not src.is_dir():
            continue
        ttfs = sorted(src.glob("*.ttf"))
        if not ttfs:
            continue
        for ttf in ttfs[:16]:
            dst = qt_fonts / ttf.name
            if dst.exists():
                continue
            try:
                dst.symlink_to(ttf)
            except OSError:
                try:
                    shutil.copy2(ttf, dst)
                except OSError:
                    pass
        break


_configure_qt_font_env()

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

import cv2

_populate_cv2_qt_fonts()

import dill
import mujoco
import mujoco.viewer
import numpy as np
import torch

from aubo_cascade_controller import AuboCascadeController, CascadeControllerConfig
from aubo_kinematics import AuboKinematics
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from keyboard_teleop_pinocchio import (
    LinuxJoystick,
    TeleopState,
    apply_joystick_motion,
    draw_scene_markers,
    format_status_line,
    get_joint_addresses,
    read_mujoco_q,
)


def render_camera(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str, image_size: int) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def preprocess_image(rgb_hwc: np.ndarray) -> np.ndarray:
    """Match AuboImageDataset: CHW float32 in [0, 1]."""
    return np.moveaxis(rgb_hwc, -1, 0).astype(np.float32) / 255.0


class ObsBuffer:
    def __init__(self, n_obs_steps: int) -> None:
        self.n_obs_steps = max(1, int(n_obs_steps))
        self._q: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)

    def push(self, frame_chw: np.ndarray) -> None:
        self._q.append(np.asarray(frame_chw, dtype=np.float32))

    def to_batch_tensor(self) -> torch.Tensor:
        if len(self._q) == 0:
            raise RuntimeError("ObsBuffer is empty")
        frames = list(self._q)
        while len(frames) < self.n_obs_steps:
            frames.insert(0, frames[0].copy())
        stacked = np.stack(frames[-self.n_obs_steps :], axis=0)
        return torch.from_numpy(stacked).unsqueeze(0)


def resolve_checkpoint_path(raw: Path) -> Path:
    """Hydra `train.py` uses cwd=third_party/diffusion_policy, so outputs are under DP_ROOT/data/outputs."""
    p = Path(raw).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([ROOT / p, DP_ROOT / p, Path.cwd() / p])

    seen: set[Path] = set()
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r in seen:
            continue
        seen.add(r)
        if r.is_file():
            return r

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Checkpoint not found: {raw}\n"
        f"Tried:\n{tried}\n\n"
        f"Training is launched from third_party/diffusion_policy, so weights are typically under:\n"
        f"  {DP_ROOT / 'data' / 'outputs'}/<date>/<run>/checkpoints/latest.ckpt"
    )


def load_policy_checkpoint(checkpoint_path: Path, device: torch.device):
    path = resolve_checkpoint_path(checkpoint_path)

    load_kw: dict = {"pickle_module": dill}
    try:
        payload = torch.load(path.open("rb"), **load_kw, weights_only=False)
    except TypeError:
        payload = torch.load(path.open("rb"), **load_kw)

    workspace = TrainDiffusionUnetImageWorkspace(payload["cfg"], output_dir=str(path.parent.parent))
    workspace.load_payload(payload)

    cfg = workspace.cfg
    policy = workspace.model
    if getattr(cfg.training, "use_ema", False) and workspace.ema_model is not None:
        policy = workspace.ema_model

    policy.eval()
    policy.to(device)

    n_obs = int(cfg.n_obs_steps)
    img_shape = tuple(cfg.task.shape_meta["obs"]["image"]["shape"])
    if len(img_shape) != 3:
        raise ValueError(f"Unexpected image shape_meta: {img_shape}")
    _, exp_h, exp_w = img_shape

    return policy, cfg, n_obs, exp_h, exp_w


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run trained Diffusion Policy in MuJoCo (AUBO sim)")
    p.add_argument("--checkpoint", "-c", required=True, type=Path, help="Path to latest.ckpt or epoch ckpt")
    p.add_argument("--device", default=None, help='Torch device (default: cuda:0 if available else cpu)')
    p.add_argument("--inference-steps", type=int, default=20, help="DDPM inference steps (lower = faster)")
    p.add_argument("--scene", default=str(root / "scene.xml"))
    p.add_argument("--urdf", default=str(root / "models" / "auboi5_mujoco_kinematics.urdf"))
    p.add_argument("--camera-name", default="ego_camera")
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--sample-rate", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=120.0)
    p.add_argument("--sim-steps", type=int, default=4)
    p.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--joystick", action="store_true", help="Enable Linux joystick teleop when policy is off")
    p.add_argument("--joystick-device", default="/dev/input/js0")
    p.add_argument("--joystick-deadzone", type=float, default=0.06)
    p.add_argument("--joystick-xyz-speed", type=float, default=0.25)
    p.add_argument("--joystick-rpy-speed", type=float, default=1.6)
    p.add_argument("--axis-x", type=int, default=0)
    p.add_argument("--axis-y", type=int, default=1)
    p.add_argument("--axis-z", type=int, default=4)
    p.add_argument("--axis-roll", type=int, default=6)
    p.add_argument("--axis-pitch", type=int, default=7)
    p.add_argument("--axis-yaw", type=int, default=3)
    p.add_argument("--invert-x", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--invert-roll", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--invert-pitch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy, _, n_obs_steps, exp_h, exp_w = load_policy_checkpoint(args.checkpoint, device)

    if args.image_size != exp_h or args.image_size != exp_w:
        raise ValueError(
            f"--image-size {args.image_size} does not match checkpoint shape_meta "
            f"{exp_h}x{exp_w}; re-run with matching resolution."
        )

    policy.num_inference_steps = max(1, int(args.inference_steps))

    mj_model = mujoco.MjModel.from_xml_path(args.scene)
    mj_data = mujoco.MjData(mj_model)
    kinematics = AuboKinematics(args.urdf)
    controller = AuboCascadeController(
        kinematics,
        CascadeControllerConfig(motion_rate=args.sample_rate, servo_rate=args.fps),
    )

    renderer = mujoco.Renderer(mj_model, height=args.image_size, width=args.image_size)
    qpos_addresses = get_joint_addresses(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    current_q = read_mujoco_q(mj_data, qpos_addresses)
    current_pose = kinematics.fk(current_q)
    state = TeleopState(target_pose=current_pose.as_vector())
    controller.reset(current_q, time.time())
    controller.set_target_pose(state.target_pose)

    obs_buffer = ObsBuffer(n_obs_steps=n_obs_steps)
    action_queue: deque[np.ndarray] = deque()

    class Ui:
        policy_enabled = False

    ui = Ui()

    joystick: LinuxJoystick | None = None
    if args.joystick:
        joystick = LinuxJoystick(args.joystick_device, axis_deadzone=args.joystick_deadzone)
        if not joystick.open():
            joystick = None

    last_loop_time = time.time()
    last_sample_time = 0.0
    dt = 1.0 / max(args.fps, 1.0)
    sample_dt = 1.0 / max(args.sample_rate, 1.0)
    last_infer_log = 0.0

    print("[keys] space: reset target to current FK | c: toggle policy | q: quit")
    print(f"[policy] device={device} inference_steps={policy.num_inference_steps} n_obs_steps={n_obs_steps}")
    print("[policy] press 'c' to enable closed-loop policy")

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except ValueError:
            state.apply_key(keycode)
            return
        if key == "c":
            ui.policy_enabled = not ui.policy_enabled
            if not ui.policy_enabled:
                action_queue.clear()
            print(f"[policy] {'ON' if ui.policy_enabled else 'OFF (manual / joystick)'}")
            return
        state.apply_key(keycode)

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running() and state.running:
            loop_start = time.time()
            loop_dt = np.clip(loop_start - last_loop_time, 1e-4, 0.05)
            last_loop_time = loop_start

            if joystick is not None and not ui.policy_enabled:
                joystick.poll(state)
                apply_joystick_motion(state, joystick, loop_dt, args)

            current_q = read_mujoco_q(mj_data, qpos_addresses)
            current_pose = kinematics.fk(current_q)
            if state.reset_requested:
                state.target_pose = current_pose.as_vector()
                state.reset_requested = False
                controller.reset(current_q, loop_start)
                action_queue.clear()

            controller.set_target_pose(state.target_pose)
            controller_state = controller.update_mujoco(mj_model, mj_data, loop_start)
            reachable_target_pose = kinematics.fk(controller_state.q_goal).as_vector()
            draw_scene_markers(viewer, reachable_target_pose)

            if loop_start - last_sample_time >= sample_dt:
                image = render_camera(renderer, mj_data, args.camera_name, args.image_size)
                obs_buffer.push(preprocess_image(image))

                if ui.policy_enabled:
                    if len(action_queue) == 0:
                        t0 = time.time()
                        with torch.no_grad():
                            batch = obs_buffer.to_batch_tensor().to(device=device, dtype=policy.dtype)
                            # Same as train workspace: obs_dict = batch['obs'] → keys match LinearNormalizer (image, …)
                            obs_dict = {"image": batch}
                            out = policy.predict_action(obs_dict)
                            chunk = out["action"][0].cpu().numpy()
                        for i in range(chunk.shape[0]):
                            action_queue.append(chunk[i].astype(np.float64, copy=False))
                        infer_ms = (time.time() - t0) * 1000.0
                        if loop_start - last_infer_log > 1.0:
                            print(
                                f"[infer] {infer_ms:.1f} ms | queued {len(action_queue)} deltas "
                                f"| {format_status_line('tar', state.target_pose)}"
                            )
                            last_infer_log = loop_start

                    if len(action_queue) > 0:
                        delta = action_queue.popleft()
                        state.apply_delta(delta)

                last_sample_time = loop_start

            if args.preview:
                disp = render_camera(renderer, mj_data, args.camera_name, args.image_size)
                frame = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
                mode = "POLICY" if ui.policy_enabled else "MANUAL"
                cv2.putText(frame, mode, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    f"q:{len(action_queue)}",
                    (6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 0),
                    1,
                )
                cv2.imshow("DP policy camera", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    state.running = False
                elif key == ord("c"):
                    ui.policy_enabled = not ui.policy_enabled
                    if not ui.policy_enabled:
                        action_queue.clear()
                    print(f"[policy] {'ON' if ui.policy_enabled else 'OFF'}")
                elif key == ord(" "):
                    state.reset_requested = True

            for _ in range(max(1, args.sim_steps)):
                mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    if joystick is not None:
        joystick.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
