#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -c "import numpy, pandas, torch" || {
  echo "Python dependencies are missing. Install CPU PyTorch, then Linux requirements:" >&2
  echo "  $PYTHON_BIN -m pip install --index-url https://download.pytorch.org/whl/cpu torch" >&2
  echo "  $PYTHON_BIN -m pip install -r requirements-linux.txt" >&2
  exit 1
}

exec "$PYTHON_BIN" "$PROJECT_ROOT/python/hrllc_tsn_trainer.py" "$@"
