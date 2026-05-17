from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="AUBO MuJoCo scene viewer")
    parser.add_argument("--scene", default=str(root / "scene.xml"), help="MuJoCo scene XML path")
    parser.add_argument("--camera", action="store_true", help="Show ego_camera preview window")
    parser.add_argument("--camera-name", default="ego_camera", help="MuJoCo camera name")
    parser.add_argument("--width", type=int, default=640, help="Camera preview width")
    parser.add_argument("--height", type=int, default=480, help="Camera preview height")
    return parser.parse_args()


class EgoCameraPreview:
    def __init__(self, model: mujoco.MjModel, camera_name: str, width: int, height: int) -> None:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera not found: {camera_name}")

        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def show(self, data: mujoco.MjData) -> bool:
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            "ego_camera preview | q: quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("MuJoCo ego_camera", bgr)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        cv2.destroyWindow("MuJoCo ego_camera")


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    camera = EgoCameraPreview(model, args.camera_name, args.width, args.height) if args.camera else None

    print("MuJoCo scene viewer started.")
    print("Use --camera to show the end-effector ego_camera preview.")
    print("Run keyboard_teleop_pinocchio.py for Pinocchio/CasADi 6D keyboard control.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            if camera is not None and not camera.show(data):
                break

            viewer.sync()
            time.sleep(model.opt.timestep)

    if camera is not None:
        camera.close()


if __name__ == "__main__":
    main()
