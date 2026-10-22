#!/bin/bash
# One-time setup of the your computer: Python environment + Jupyter kernel "MLAIDP (Python 3.12)".
# Needs uv (brew install uv). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
uv venv -q --allow-existing --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name mlaidp --display-name "MLAIDP (Python 3.12)" >/dev/null
.venv/bin/python tools/build.py
echo "Environment ready. Open notebooks/ with the kernel 'MLAIDP (Python 3.12)' (AWS credentials: ${AWS_PROFILE:-default chain})."
