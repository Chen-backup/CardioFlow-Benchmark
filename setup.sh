#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

if [ -n "${PIP_INDEX_URL}" ]; then
  python -m pip install -r requirements.txt -i "${PIP_INDEX_URL}"
else
  python -m pip install -r requirements.txt
fi

python - <<'PY'
import torch
print("Installation complete")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
