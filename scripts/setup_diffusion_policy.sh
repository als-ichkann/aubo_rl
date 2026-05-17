#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-aubo_rl}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DP_DIR="${ROOT_DIR}/third_party/diffusion_policy"
INSTALL_DP_EDITABLE="${INSTALL_DP_EDITABLE:-1}"

echo "[setup] Project root: ${ROOT_DIR}"
echo "[setup] Target conda env: ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda is not available in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[setup] Python: $(python --version)"
echo "[setup] You may need system packages:"
echo "        sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf"

if [ ! -d "${DP_DIR}" ]; then
  cat >&2 <<EOF
[error] diffusion_policy source directory was not found:
  ${DP_DIR}

This script no longer clones diffusion_policy automatically.
Please place your manually installed source at the path above, or edit DP_DIR in this script.
EOF
  exit 1
fi
echo "[setup] Using existing diffusion_policy source: ${DP_DIR}"

echo "[setup] Installing Python dependencies"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"
python -m pip install \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
echo "[setup] Checking NCCL runtime library"
if ! python - <<'PY'
import site
from pathlib import Path

for site_dir in site.getsitepackages():
    if list((Path(site_dir) / "nvidia").glob("**/libnccl.so.2")):
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  echo "[setup] libnccl.so.2 is missing; reinstalling nvidia-nccl-cu12"
  python -m pip install --force-reinstall --no-cache-dir nvidia-nccl-cu12==2.20.5
fi
python -m pip install \
  "zarr<3" "numcodecs<0.16" \
  hydra-core omegaconf einops diffusers dill wandb tqdm numba \
  imageio imageio-ffmpeg av termcolor scikit-image scikit-video pandas \
  opencv-python h5py robomimic unidiff

if [ "${INSTALL_DP_EDITABLE}" = "1" ]; then
  echo "[setup] Installing existing diffusion_policy in editable mode"
  python -m pip install -e "${DP_DIR}"
else
  echo "[setup] Skipping editable diffusion_policy install"
fi

echo "[setup] Ensuring diffusion_policy source is on PYTHONPATH via .pth"
DP_DIR="${DP_DIR}" python - <<'PY'
import os
import site
from pathlib import Path

dp_dir = Path(os.environ["DP_DIR"]).resolve()
site_dir = Path(site.getsitepackages()[0])
pth_path = site_dir / "aubo_diffusion_policy_source.pth"
pth_path.write_text(str(dp_dir) + "\n", encoding="utf-8")
print(f"ok: wrote {pth_path} -> {dp_dir}")
PY

echo "[setup] Verifying imports"
python - <<'PY'
import importlib
for name in ["mujoco", "zarr", "torch", "diffusion_policy"]:
    importlib.import_module(name)
    print(f"ok: {name}")
PY

echo "[setup] Done."
