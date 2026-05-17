#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-aubo_rl}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DP_DIR="${ROOT_DIR}/third_party/diffusion_policy"
CONFIG_DIR="${ROOT_DIR}/configs/diffusion_policy"
CONFIG_NAME="${CONFIG_NAME:-aubo_ego_rgb_delta_pose}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda is not available in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

if [ ! -d "${DP_DIR}" ]; then
  echo "[error] diffusion_policy source not found at ${DP_DIR}" >&2
  echo "        Run scripts/setup_diffusion_policy.sh first." >&2
  exit 1
fi

if [ ! -f "${CONFIG_DIR}/${CONFIG_NAME}.yaml" ]; then
  echo "[error] config not found: ${CONFIG_DIR}/${CONFIG_NAME}.yaml" >&2
  exit 1
fi

cd "${DP_DIR}"
echo "[train] config-dir=${CONFIG_DIR}"
echo "[train] config-name=${CONFIG_NAME}"
python train.py --config-dir="${CONFIG_DIR}" --config-name="${CONFIG_NAME}" "$@"
