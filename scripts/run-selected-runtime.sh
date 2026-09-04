#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -lt 5 || "$1" != "--python" || "$3" != "--runtime-selection" ]]; then
    exit 2
fi
SELECTED_PYTHON="$2"
RUNTIME_SELECTION="$4"
shift 4

unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS
unset PYTHONBREAKPOINT PYTHONEXECUTABLE PYTHONPLATLIBDIR VIRTUAL_ENV CONDA_PREFIX
unset _PYTHON_SYSCONFIGDATA_NAME __PYVENV_LAUNCHER__
export PYTHONPATH="${ROOT_DIR}/src"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
exec "${SELECTED_PYTHON}" -P -m fun_voice.runtime_launcher \
    --runtime-selection "${RUNTIME_SELECTION}" "$@"
