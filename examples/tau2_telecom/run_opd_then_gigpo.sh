#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${OPD_HF_MODEL:-}" ]]; then
  echo "Set OPD_HF_MODEL to the exported hf_model directory from the OPD checkpoint." >&2
  echo "This script intentionally starts GiGPO with a fresh optimizer instead of resuming the OPD FSDP state." >&2
  exit 1
fi
if [[ ! -d "$OPD_HF_MODEL" ]]; then
  echo "OPD_HF_MODEL is not a directory: $OPD_HF_MODEL" >&2
  exit 1
fi

export STUDENT_MODEL="$OPD_HF_MODEL"
export EXP_NAME="${EXP_NAME:-qwen3_0p6b_opd_then_gigpo}"
exec bash "$SCRIPT_DIR/run_gigpo.sh" "$@"

