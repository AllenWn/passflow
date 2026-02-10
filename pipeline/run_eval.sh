#!/usr/bin/env bash
set -euo pipefail

# Make paths robust no matter where invoked from.
ORIG_PWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

#   bash run_eval.sh
#   bash run_eval.sh config.py
#   bash run_eval.sh path/to/config.py
#   bash run_eval.sh --batch [config.py]

if [[ "${1:-}" == "--batch" ]]; then
  CONFIG_PATH="${2:-config.py}"
  CONFIG_PATH="$(cd "${ORIG_PWD}" && python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "${CONFIG_PATH}")"
  echo "[RUN] Batch mode with config: ${CONFIG_PATH}"
  echo

  python3 run_batch.py --config "${CONFIG_PATH}"

  exit 0
fi

CONFIG_PATH="${1:-config.py}"
CONFIG_PATH="$(cd "${ORIG_PWD}" && python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "${CONFIG_PATH}")"

echo "[RUN] Using config: ${CONFIG_PATH}"
echo

# dry run check output directory
OUTDIR=$(
  python3 run_generate.py --config "${CONFIG_PATH}" --dry_run \
  | awk -F': ' '/^\[INFO\] Output dir:/ {print $2; exit}'
)

if [[ -z "${OUTDIR}" ]]; then
  echo "[ERROR] Failed to parse output directory from dry_run output."
  exit 1
fi

SAMPLES_PATH="${OUTDIR}/samples.jsonl"
EVAL_LOG="${OUTDIR}/eval.log"

echo "[INFO] Parsed outdir: ${OUTDIR}"
echo "[INFO] Samples will be: ${SAMPLES_PATH}"
echo

# run generation
echo "[STEP 1/2] Generating samples via vLLM..."
python3 run_generate.py --config "${CONFIG_PATH}"


RUNCFG_PATH="${OUTDIR}/run_config.json"

# run evaluation
# normalize benchmark name for evalplus.evaluate positional argument
BENCHMARK=$(
  python3 - "${RUNCFG_PATH}" << 'EOF'
import json, sys
cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
b = cfg["benchmark"].lower()
if b in ("humaneval", "humaneval+"):
    b = "humaneval"
elif b in ("mbpp", "mbpp+"):
    b = "mbpp"
print(b)
EOF
)

echo
echo "[STEP 2/2] Evaluating with EvalPlus..."
echo "[INFO] Command: evalplus.evaluate ${BENCHMARK} --samples ${SAMPLES_PATH}"
echo "[INFO] Logging to: ${EVAL_LOG}"
echo

# tee output to eval.log
evalplus.evaluate "${BENCHMARK}" --samples "${SAMPLES_PATH}" | tee "${EVAL_LOG}"

echo
echo "[DONE] Results saved:"
echo "  - ${SAMPLES_PATH}"
echo "  - ${OUTDIR}/run_config.json"
echo "  - ${EVAL_LOG}"
