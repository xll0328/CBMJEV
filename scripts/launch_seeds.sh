#!/usr/bin/env bash
# Explicit task-level scheduling only; never consumes the experiment matrix.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG-:4096:8}"

usage() {
  printf '%s\n' 'Usage: bash scripts/launch_seeds.sh --prepared DIR --out NEW_DIR --seeds 17,18,19 [options]' \
    '  --gpus 0,1,2      One distinct GPU per explicitly listed seed; up to 8 jobs' \
    '                    Without --gpus, selected seeds run sequentially on CPU' \
    '  --config FILE     Training JSON (default: configs/cebab_cpu.json)' \
    '  --epochs N        Responder epochs only (default: 30)' \
    '  --python EXE      Python executable (default: python3)' \
    '  --live            Include validation live rollouts; parallel timing is not paper latency' \
    '  --dry-run         Validate every job and print commands; create nothing' \
    '  --help            Show this help' \
    'No default seeds, downloads, test evaluation, calibration, or matrix expansion.'
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "missing value for $1"; }
absolute() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }
CBMJEV_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
prepared=''; out=''; seed_csv=''; gpu_csv=''; config="$CBMJEV_SCRIPT_DIR/../configs/cebab_cpu.json"
epochs=30; python_exe=python3; live=0; dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepared) need_value "$@"; prepared=$(absolute "$2"); shift 2 ;;
    --out) need_value "$@"; out=$(absolute "$2"); shift 2 ;;
    --seeds) need_value "$@"; seed_csv=$2; shift 2 ;;
    --gpus) need_value "$@"; gpu_csv=$2; shift 2 ;;
    --config) need_value "$@"; config=$(absolute "$2"); shift 2 ;;
    --epochs) need_value "$@"; epochs=$2; shift 2 ;;
    --python) need_value "$@"; python_exe=$2; shift 2 ;;
    --live) live=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done
[[ -n "$prepared" && -n "$out" && -n "$seed_csv" ]] || fail '--prepared, --out and explicit --seeds are required'
[[ ! -e "$out" && ! -L "$out" ]] || fail "output already exists: $out"
[[ "$seed_csv" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || fail 'seeds must be comma-separated nonnegative integers'
IFS=',' read -r -a seeds <<< "$seed_csv"
[[ "${#seeds[@]}" -le 8 ]] || fail 'at most eight explicit jobs are supported'
gpus=()
if [[ -n "$gpu_csv" ]]; then
  [[ "$gpu_csv" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || fail 'gpus must be comma-separated physical indices'
  IFS=',' read -r -a gpus <<< "$gpu_csv"
  [[ "${#gpus[@]}" -eq "${#seeds[@]}" ]] || fail 'provide exactly one GPU per seed'
fi
for ((i=0; i<${#seeds[@]}; i++)); do
  for ((j=0; j<i; j++)); do
    [[ "${seeds[i]}" != "${seeds[j]}" ]] || fail 'duplicate seeds are not allowed'
    if [[ -n "$gpu_csv" ]]; then [[ "${gpus[i]}" != "${gpus[j]}" ]] || fail 'duplicate GPU assignments are not allowed'; fi
  done
done
job() (
  local index=$1
  shift
  local args=(--prepared "$prepared" --out "$out/seed_${seeds[index]}" --config "$config" \
    --seed "${seeds[index]}" --epochs "$epochs" --python "$python_exe")
  if [[ -n "$gpu_csv" ]]; then args+=(--device cuda:0 --gpu "${gpus[index]}"); fi
  if [[ "$live" -eq 1 ]]; then args+=(--live); fi
  job_pid=''
  trap 'if [[ -n "$job_pid" ]]; then kill -TERM "$job_pid" 2>/dev/null || true; wait "$job_pid" 2>/dev/null || true; fi; exit 130' INT TERM
  bash "$CBMJEV_SCRIPT_DIR/run_cebab.sh" "${args[@]}" "$@" &
  job_pid=$!
  wait "$job_pid"
)
# Validate all jobs before creating the parent or starting any training process.
for ((i=0; i<${#seeds[@]}; i++)); do job "$i" --dry-run; done
if [[ "$dry_run" -eq 1 ]]; then exit 0; fi
[[ -d "$(dirname "$out")" ]] || fail 'output parent must already exist'
mkdir "$out"
pids=()
interrupt_jobs() {
  trap - INT TERM
  if [[ "${#pids[@]}" -gt 0 ]]; then
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
  printf 'Interrupted. Partial output retained at %s; rerun into a new directory.\n' "$out" >&2
  exit 130
}
trap interrupt_jobs INT TERM
if [[ -z "$gpu_csv" ]]; then
  for ((i=0; i<${#seeds[@]}; i++)); do
    job "$i" >"$out/seed_${seeds[i]}.log" 2>&1 &
    pids=("$!")
    wait "${pids[0]}"
    pids=()
  done
else
  for ((i=0; i<${#seeds[@]}; i++)); do
    job "$i" >"$out/seed_${seeds[i]}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  [[ "$failed" -eq 0 ]] || fail "one or more jobs failed; logs and partial output retained in $out"
fi
printf 'Completed %s explicitly selected seeds. Separate R/cache per seed; test not evaluated.\n' "${#seeds[@]}"
