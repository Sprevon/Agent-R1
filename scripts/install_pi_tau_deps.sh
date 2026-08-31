#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIDECAR_DIR="$PROJECT_DIR/recipes/tau2_telecom/pi_sidecar"
PYTHON_BIN="${PYTHON_BIN:-python3}"

node -e 'const [major, minor] = process.versions.node.split(".").map(Number); if (major < 22 || (major === 22 && minor < 19)) process.exit(1)' || {
  echo "Node >=22.19.0 is required by pi-agent-core 0.84.4" >&2
  exit 1
}

npm --prefix "$SIDECAR_DIR" install --ignore-scripts
"$PYTHON_BIN" -m pip install -r "$PROJECT_DIR/requirements-tau2.txt"
"$PYTHON_BIN" -m pip install -r "$PROJECT_DIR/requirements-opd.txt" --no-deps

npm --prefix "$SIDECAR_DIR" run check
echo "Pi/tau2/OPD dependencies installed. Run scripts/check_pi_tau_environment.py next."

