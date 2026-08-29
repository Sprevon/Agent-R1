#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-all}"
if [[ "$PHASE" == "opd" || "$PHASE" == "gigpo" || "$PHASE" == "all" ]]; then
  if [[ $# -gt 0 ]]; then shift; fi
else
  echo "Usage: $0 [opd|gigpo|all] [hydra overrides...]" >&2
  exit 2
fi

export STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-0.6B}"
export TAU2_TRAIN_BATCH_SIZE="${TAU2_TRAIN_BATCH_SIZE:-1}"
export TAU2_PPO_MINI_BATCH_SIZE="${TAU2_PPO_MINI_BATCH_SIZE:-1}"
export AGENT_FLOW_WORKERS="${AGENT_FLOW_WORKERS:-1}"
export PI_TAU_MAX_STEPS="${PI_TAU_MAX_STEPS:-4}"
export TAU2_GIGPO_ROLLOUT_N="${TAU2_GIGPO_ROLLOUT_N:-2}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export TEST_FREQ="${TEST_FREQ:-1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export EXP_NAME="${EXP_NAME:-tau2_telecom_3080_smoke}"
export TAU2_TRAIN_PATH="${TAU2_TRAIN_PATH:-$SCRIPT_DIR/../../data/tau2_telecom_smoke/train.parquet}"
export TAU2_VAL_PATH="${TAU2_VAL_PATH:-$SCRIPT_DIR/../../data/tau2_telecom_smoke/test.parquet}"

SCRIPT_PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" "$SCRIPT_PROJECT_DIR/scripts/check_pi_tau_environment.py" \
  --student "$STUDENT_MODEL" \
  --teacher "$TEACHER_MODEL"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  if [[ "$PHASE" != "gigpo" && "$GPU_COUNT" -lt 2 ]]; then
    echo "The pinned OPD runtime uses a dedicated teacher GPU pool and needs two visible GPUs." >&2
    echo "A one-GPU host can run the Pi/tau2 checks, but not the full OPD optimizer-update smoke." >&2
    exit 1
  fi
fi

if [[ "$PHASE" == "opd" || "$PHASE" == "all" ]]; then
  bash "$SCRIPT_DIR/run_opd.sh" "$@"
fi
if [[ "$PHASE" == "gigpo" || "$PHASE" == "all" ]]; then
  bash "$SCRIPT_DIR/run_gigpo.sh" "$@"
fi
