#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_conda.sh
#   bash setup_conda.sh -n myenv
#   bash setup_conda.sh -n myenv -p 3.12


ENV_NAME="evalplus-pipeline"
PY_VER="3.12"

while getopts "n:p:h" opt; do
  case "${opt}" in
    n) ENV_NAME="${OPTARG}" ;;
    p) PY_VER="${OPTARG}" ;;
    h)
      echo "Usage: bash setup_conda.sh [-n env_name] [-p python_version]"
      exit 0
      ;;
    *)
      echo "Unknown option. Use -h for help."
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
  echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[INFO] Creating conda env: ${ENV_NAME} (python=${PY_VER})"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Env already exists: ${ENV_NAME} (will reuse)"
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}" pip
fi

conda activate "${ENV_NAME}"

echo "[INFO] Upgrading pip tooling..."
python -m pip install -U pip setuptools wheel

echo "[INFO] Installing EvalPlus + vLLM backend..."
# Latest stable from PyPI:
python -m pip install -U "evalplus[vllm]"

echo "[INFO] Sanity check imports..."
python - <<'PY'
import sys
print("python:", sys.version)
import evalplus
print("evalplus:", getattr(evalplus, "__version__", "unknown"))
import vllm
print("vllm:", getattr(vllm, "__version__", "unknown"))
PY

echo
echo "[OK] Environment ready."
echo "Next:"
echo "  conda activate ${ENV_NAME}"
echo "  cd \"$(pwd)\""
# echo "  cd pipeline"
# echo "  bash run_eval.sh config.py"
# echo "  bash run_eval.sh --batch config.py"

