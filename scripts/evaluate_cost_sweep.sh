#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_cost_sweep.sh --models DIR --cache DIR --out DIR \
  --method risk|value|nano_risk --gpu INDEX [--weights "0 0.01 0.03 0.1"] \
  [--budgets "0 1 2 3 4"] [--python PYTHON] [--stopping-controls] \
  [--nano-controller DIR --matched-controls]

Runs validation-only offline replay. Every weight gets an immutable subdirectory.
EOF
}
fail() { printf '%s\n' "$*" >&2; exit 2; }
models=''; cache=''; out=''; method=''; gpu=''; weights='0 0.005 0.01 0.02 0.03 0.05 0.1 0.2'; budgets=''; python_exe=python3
stopping_controls=false
nano_controller=''; matched_controls=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) models=$2; shift 2 ;;
    --cache) cache=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --method) method=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --weights) weights=$2; shift 2 ;;
    --budgets) budgets=$2; shift 2 ;;
    --python) python_exe=$2; shift 2 ;;
    --stopping-controls) stopping_controls=true; shift ;;
    --nano-controller) nano_controller=$2; shift 2 ;;
    --matched-controls) matched_controls=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done
[[ -d "$models" && -d "$cache" ]] || fail 'models/cache directories are required'
[[ -n "$out" && ! -e "$out" ]] || fail 'out must be a new path'
[[ "$method" == risk || "$method" == value || "$method" == nano_risk ]] || fail 'method must be risk, value or nano_risk'
[[ "$stopping_controls" == false || "$method" == value ]] || fail 'stopping controls require value objective'
[[ "$matched_controls" == false || "$method" == nano_risk ]] || fail 'matched controls require nano_risk'
if [[ "$method" == nano_risk ]]; then
  [[ -f "$nano_controller/training.json" ]] || fail 'nano_risk requires a completed --nano-controller'
elif [[ -n "$nano_controller" ]]; then
  fail '--nano-controller requires nano_risk'
fi
[[ "$gpu" =~ ^(0|1)$ ]] || fail 'this project currently authorizes physical GPU 0 or 1 only'
command -v "$python_exe" >/dev/null 2>&1 || fail 'python executable unavailable'
command -v nvidia-smi >/dev/null 2>&1 || fail 'nvidia-smi unavailable'
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ "$gpu_uuid" == GPU-* ]] || fail 'could not resolve GPU UUID'
mkdir "$out"
run_dirs=()
num_groups=$("$python_exe" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["groups"]))' "$models/schema.json")
budget_values=${budgets:-$num_groups}
for budget in $budget_values; do
  [[ "$budget" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'budgets must be nonnegative integers'
  (( budget <= num_groups )) || fail 'budget exceeds schema query groups'
  for weight in $weights; do
    "$python_exe" -c 'import math,sys; x=float(sys.argv[1]); assert math.isfinite(x) and x >= 0' "$weight"
    tag=${weight//./p}
    run_dir="$out/budget_${budget}_cost_${tag}"
    methods=(stop fixed random static "$method")
    if (( budget == num_groups )); then methods=(stop all fixed random static "$method"); fi
    if [[ "$stopping_controls" == true ]]; then methods+=(static_value value_singleton); fi
    extra_args=()
    if [[ "$method" == nano_risk ]]; then extra_args+=(--nano-controller "$nano_controller"); fi
    if [[ "$matched_controls" == true ]]; then methods+=(matched_mlp_risk); fi
    env CUDA_VISIBLE_DEVICES="$gpu_uuid" "$python_exe" "$project_dir/tools/run_gpu_limited.py" \
      --memory-gib 4 --reserve-gib 8 -- -m cbmjev evaluate \
      --models "$models" --cache "$cache" --out "$run_dir" --device cuda:0 \
      --methods "${methods[@]}" --cost-weight "$weight" --max-groups "$budget" "${extra_args[@]}"
    run_dirs+=("$run_dir")
  done
done
"$python_exe" "$project_dir/tools/summarize_cost_sweep.py" --runs "${run_dirs[@]}" --out "$out/summary"
