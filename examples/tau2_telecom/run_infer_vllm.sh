#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PATH="${PI_NODE_BIN_DIR:-}${PI_NODE_BIN_DIR:+:}${PATH:-}"
export TAU2_SOLO_MODE="${TAU2_SOLO_MODE:-true}"

MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${TAU2_INFER_OUTPUT_DIR:-$PROJECT_DIR/artifacts/tau2_telecom_infer}"
DATA_FILES=("${TAU2_INFER_DATA:-$PROJECT_DIR/data/tau2_telecom_smoke/train.parquet}")
if [[ -z "${TAU2_INFER_DATA:-}" && -f "$PROJECT_DIR/data/tau2_telecom_smoke/test.parquet" ]]; then
  DATA_FILES+=("$PROJECT_DIR/data/tau2_telecom_smoke/test.parquet")
fi

exec "$PYTHON_BIN" -m recipes.tau2_telecom.infer_vllm_pi \
  --model "$MODEL" \
  --data "${DATA_FILES[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --max_tasks "${TAU2_INFER_MAX_TASKS:-2}" \
  --max_turns "${PI_TAU_MAX_STEPS:-8}" \
  --max_tokens "${TAU2_MAX_RESPONSE_LEN:-512}" \
  --max_prompt_len "${TAU2_MAX_PROMPT_LEN:-12288}" \
  --max_model_len "${TAU2_MAX_MODEL_LEN:-12800}" \
  --temperature "${TAU2_INFER_TEMPERATURE:-0.0}" \
  --gpu_memory_utilization "${TAU2_INFER_GPU_MEMORY_UTILIZATION:-0.70}" \
  --solo_mode "${TAU2_SOLO_MODE:-true}" \
  "$@"
