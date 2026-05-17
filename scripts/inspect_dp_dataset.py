from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect AUBO diffusion policy zarr dataset")
    parser.add_argument("dataset", nargs="?", default="expert_demos/aubo_ego_rgb_delta_pose.zarr")
    parser.add_argument("--save-dir", default="expert_demos/preview_frames")
    parser.add_argument("--num-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    root = zarr.open_group(str(dataset_path), mode="r")
    image = root["data/image"]
    action = root["data/action"]
    eef_pose = root["data/eef_pose"]
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)

    print(f"dataset: {dataset_path}")
    print(f"episodes: {len(episode_ends)}")
    print(f"steps: {action.shape[0]}")
    print(f"image: shape={image.shape}, dtype={image.dtype}")
    print(f"action: shape={action.shape}, dtype={action.dtype}")
    print(f"eef_pose: shape={eef_pose.shape}, dtype={eef_pose.dtype}")
    print(f"episode_ends: {episode_ends.tolist()}")

    if action.shape[0] > 0:
        action_np = np.asarray(action[:], dtype=np.float32)
        print(f"action mean: {np.round(action_np.mean(axis=0), 5)}")
        print(f"action std: {np.round(action_np.std(axis=0), 5)}")
        print(f"action min: {np.round(action_np.min(axis=0), 5)}")
        print(f"action max: {np.round(action_np.max(axis=0), 5)}")

    if image.shape[0] == 0 or args.num_frames <= 0:
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, image.shape[0] - 1, min(args.num_frames, image.shape[0]), dtype=int)
    for idx in indices:
        rgb = np.asarray(image[idx], dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_path = save_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(out_path), bgr)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
