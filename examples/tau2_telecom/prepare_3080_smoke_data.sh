#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m recipes.tau2_telecom.data_preprocess.process_tau2_telecom \
  --output_dir data/tau2_telecom_smoke \
  --splits train test \
  --max_tasks 1 \
  --seed 42
