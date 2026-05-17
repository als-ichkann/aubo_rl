#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-aubo_rl}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DP_DIR="${ROOT_DIR}/third_party/diffusion_policy"
DP_REPO="${DP_REPO:-https://github.com/real-stanford/diffusion_policy.git}"

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

mkdir -p "${ROOT_DIR}/third_party"
if [ ! -d "${DP_DIR}/.git" ]; then
  echo "[setup] Cloning diffusion_policy from ${DP_REPO}"
  git clone "${DP_REPO}" "${DP_DIR}"
else
  echo "[setup] diffusion_policy already exists, pulling latest changes"
  git -C "${DP_DIR}" pull --ff-only || true
fi

echo "[setup] Installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch" "torchvision" \
  "zarr<3" "numcodecs<0.16" \
  hydra-core omegaconf einops diffusers dill wandb tqdm \
  imageio imageio-ffmpeg av termcolor scikit-image scikit-video pandas \
  opencv-python h5py robomimic

echo "[setup] Installing diffusion_policy in editable mode"
python -m pip install -e "${DP_DIR}"

echo "[setup] Verifying imports"
python - <<'PY'
import importlib
for name in ["mujoco", "zarr", "torch", "diffusion_policy"]:
    importlib.import_module(name)
    print(f"ok: {name}")
PY

echo "[setup] Done."
