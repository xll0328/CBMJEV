#!/usr/bin/env bash
# Thin entry point: all planning, resource guards and lifecycle receipts are Python.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_exe=python3
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --python)
      (( index + 1 < ${#arguments[@]} )) || { printf '%s\n' 'missing value for --python' >&2; exit 2; }
      python_exe=${arguments[index+1]}
      ;;
    --python=*) python_exe=${arguments[index]#--python=} ;;
  esac
done
command -v "$python_exe" >/dev/null 2>&1 || { printf 'Python unavailable: %s\n' "$python_exe" >&2; exit 2; }
exec "$python_exe" "$script_dir/run_vision.py" "$@"
