from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import mujoco
import numpy as np
import zarr


def check_imports(require_diffusion_policy: bool) -> None:
    modules = ["mujoco", "zarr", "cv2", "numpy"]
    if require_diffusion_policy:
        modules.append("diffusion_policy")
    for module in modules:
        importlib.import_module(module)
        print(f"ok import: {module}")


def check_scene(scene_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    print(f"ok scene: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    if model.nu != 6:
        raise RuntimeError(f"Expected 6 actuators, got {model.nu}")


def check_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        print(f"skip dataset: {dataset_path} does not exist yet")
        return

    root = zarr.open_group(str(dataset_path), mode="r")
    image = root["data/image"]
    action = root["data/action"]
    eef_pose = root["data/eef_pose"]
    episode_ends = np.asarray(root["meta/episode_ends"][:])
    print(f"ok dataset: steps={action.shape[0]}, episodes={episode_ends.shape[0]}")
    print(f"image shape={image.shape}, action shape={action.shape}, eef_pose shape={eef_pose.shape}")
    if image.shape[0] != action.shape[0] or action.shape[0] != eef_pose.shape[0]:
        raise RuntimeError("Dataset arrays have inconsistent first dimensions")
    if action.shape[-1] != 6:
        raise RuntimeError(f"Expected action dim 6, got {action.shape[-1]}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Smoke test AUBO diffusion policy data pipeline")
    parser.add_argument("--scene", default=str(root / "scene.xml"))
    parser.add_argument("--dataset", default=str(root / "expert_demos" / "aubo_ego_rgb_delta_pose.zarr"))
    parser.add_argument("--config", default=str(root / "configs" / "diffusion_policy" / "aubo_ego_rgb_delta_pose.yaml"))
    parser.add_argument("--require-diffusion-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_imports(args.require_diffusion_policy)
    check_scene(Path(args.scene))
    config_path = Path(args.config)
    if not config_path.exists():
        raise RuntimeError(f"Missing config: {config_path}")
    print(f"ok config: {config_path}")
    check_dataset(Path(args.dataset))
    print("smoke test ok")


if __name__ == "__main__":
    main()
