#!/usr/bin/env bash
set -euo pipefail

# Cleanup function to remove environment if setup fails
cleanup_on_error() {
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo ""
    echo "[ERROR] Setup failed with exit code $exit_code"
    echo "[INFO] Cleaning up partial environment: ${ENV_NAME}"
    conda env remove -y -n "${ENV_NAME}" 2>/dev/null || true
    echo "[INFO] Please try again or run: conda env remove -y -n ${ENV_NAME}"
  fi
  exit $exit_code
}
trap cleanup_on_error EXIT

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
# Use conda shell.bash hook for better cross-platform compatibility
eval "$(conda shell.bash hook)"

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
# Try to install evalplus with vllm support
if python -m pip install -U "evalplus[vllm]"; then
  echo "[OK] EvalPlus + vLLM installed successfully"
else
  echo "[WARN] vLLM installation failed (may require C++ build tools on some systems)"
  echo "[INFO] Retrying with evalplus only..."
  python -m pip install -U "evalplus" || { echo "[ERROR] Failed to install evalplus"; exit 1; }
  echo "[INFO] You may need to install vLLM separately or skip it for this environment"
fi

echo "[INFO] Sanity check imports..."
python - <<'PY'
import sys
print("python:", sys.version)
import evalplus
print("evalplus:", getattr(evalplus, "__version__", "unknown"))
try:
  import vllm
  print("vllm:", getattr(vllm, "__version__", "unknown"))
except ImportError:
  print("vllm: NOT INSTALLED (optional)")
PY

echo
echo "[OK] Environment ready."
echo "Next:"
echo "  conda activate ${ENV_NAME}"
echo "  cd \"$(pwd)\""
# echo "  cd pipeline"
# echo "  bash run_eval.sh config.py"
# echo "  bash run_eval.sh --batch config.py"

# Disable cleanup trap on successful completion
trap - EXIT