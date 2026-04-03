#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m ensurepip --upgrade
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install \
  numpy \
  faiss-cpu \
  openai \
  pytest \
  datasets \
  ragas \
  langchain-openai \
  langchain-ollama

cat <<EOF
Environment initialized.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Run tests with:
  "${VENV_DIR}/bin/python" -m pytest benchmark/tests -q

Start the app with:
  "${VENV_DIR}/bin/python" -m streamlit run app.py
EOF
