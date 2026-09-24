#!/usr/bin/env bash
# Explicit, local-data-only validation chain. No download, test, or certification.
set -euo pipefail
# Required by deterministic CUDA linear algebra; preserve an explicit user setting.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG-:4096:8}"

usage() {
  printf '%s\n' 'Usage: bash scripts/run_cebab.sh --prepared DIR --out NEW_DIR [options]' \
    '  --config FILE     Training JSON (default: configs/cebab_cpu.json)' \
    '  --seed N          Training seed, not data split seed (default: 17)' \
    '  --device DEVICE   cpu (default) or cuda:0 when using --gpu' \
    '  --gpu N           One physical GPU index; scoped to this job only' \
    '  --gpu-memory-gib N  Optional positive allocator budget; requires cuda:0 + --gpu' \
    '  --gpu-reserve-gib N Free-memory headroom for limited runs (default: 8)' \
    '  --epochs N        Responder epochs only (default: 30)' \
    '  --responder-kind K hashing_text (default) or hf_text' \
    '  --backbone SOURCE  Required local/HF encoder source for hf_text' \
    '  --revision SHA     Optional immutable HF revision for hf_text' \
    '  --allow-download   Explicitly permit hf_text Hub download' \
    '  --max-length N     hf_text token limit (default: 512)' \
    '  --freeze-backbone  Train only hf_text concept heads' \
    '  --python EXE      Python executable (default: python3)' \
    '  --live            Also run live validation; not an isolated speed benchmark' \
    '  --dry-run         Validate paths/config and print commands; write nothing' \
    '  --help            Show this help' \
    'Always fits a separate semantic responder and cache. Never evaluates test.'
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "missing value for $1"; }
absolute() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }

CBMJEV_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CBMJEV_PROJECT_DIR=$(cd "$CBMJEV_SCRIPT_DIR/.." && pwd)
prepared=''; out=''; config="$CBMJEV_PROJECT_DIR/configs/cebab_cpu.json"
seed=17; device=cpu; gpu=''; gpu_memory_gib=''; gpu_reserve_gib=8
epochs=30; python_exe=python3; live=0; dry_run=0
responder_kind=hashing_text; backbone=''; revision=''; allow_download=0; max_length=512; freeze_backbone=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepared) need_value "$@"; prepared=$(absolute "$2"); shift 2 ;;
    --out) need_value "$@"; out=$(absolute "$2"); shift 2 ;;
    --config) need_value "$@"; config=$(absolute "$2"); shift 2 ;;
    --seed) need_value "$@"; seed=$2; shift 2 ;;
    --device) need_value "$@"; device=$2; shift 2 ;;
    --gpu) need_value "$@"; gpu=$2; shift 2 ;;
    --gpu-memory-gib) need_value "$@"; gpu_memory_gib=$2; shift 2 ;;
    --gpu-reserve-gib) need_value "$@"; gpu_reserve_gib=$2; shift 2 ;;
    --epochs) need_value "$@"; epochs=$2; shift 2 ;;
    --responder-kind) need_value "$@"; responder_kind=$2; shift 2 ;;
    --backbone) need_value "$@"; backbone=$2; shift 2 ;;
    --revision) need_value "$@"; revision=$2; shift 2 ;;
    --allow-download) allow_download=1; shift ;;
    --max-length) need_value "$@"; max_length=$2; shift 2 ;;
    --freeze-backbone) freeze_backbone=1; shift ;;
    --python) need_value "$@"; python_exe=$2; shift 2 ;;
    --live) live=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done
[[ -n "$prepared" && -n "$out" ]] || fail '--prepared and --out are required'
[[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'seed must be a nonnegative integer'
[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || fail 'epochs must be a positive integer'
[[ "$max_length" =~ ^[1-9][0-9]*$ ]] || fail 'max-length must be a positive integer'
[[ "$responder_kind" == hashing_text || "$responder_kind" == hf_text ]] || fail 'responder-kind must be hashing_text or hf_text'
if [[ "$responder_kind" == hf_text ]]; then
  [[ -n "$backbone" ]] || fail 'hf_text requires --backbone'
else
  [[ -z "$backbone" && -z "$revision" && "$allow_download" -eq 0 && "$freeze_backbone" -eq 0 && "$max_length" -eq 512 ]] \
    || fail 'HF responder options require --responder-kind hf_text'
fi
[[ "$device" == cpu || "$device" == cuda:0 ]] || fail 'device must be cpu or cuda:0'
if [[ -n "$gpu" ]]; then
  [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'gpu must be one nonnegative physical index'
  [[ "$device" == cuda:0 ]] || fail '--gpu requires explicit --device cuda:0'
elif [[ "$device" != cpu ]]; then
  fail 'CUDA runs require an explicit --gpu; no implicit use of all GPUs'
fi
if [[ -n "$gpu_memory_gib" ]]; then
  [[ "$device" == cuda:0 && -n "$gpu" ]] || fail '--gpu-memory-gib requires --device cuda:0 and --gpu'
  [[ -f "$CBMJEV_PROJECT_DIR/tools/run_gpu_limited.py" ]] || fail 'GPU-limited runner is missing'
fi
[[ ! -e "$out" && ! -L "$out" ]] || fail "output already exists: $out"
[[ -d "$prepared" && -f "$config" ]] || fail 'prepared directory or config is missing'
for name in schema.json samples.jsonl membership.jsonl; do
  [[ -f "$prepared/$name" ]] || fail "missing prepared artifact: $name"
done
command -v "$python_exe" >/dev/null 2>&1 || fail "Python executable unavailable: $python_exe"
if [[ "$python_exe" == */* ]]; then python_exe=$(absolute "$python_exe"); fi
cd "$CBMJEV_PROJECT_DIR"
# Parse JSON and validate the typed configuration before creating output or training.
env PYTHONDONTWRITEBYTECODE=1 "$python_exe" -c '
import json,math,sys
from cbmjev.config import resolve_config
for name, raw in (("gpu-memory-gib", sys.argv[3]), ("gpu-reserve-gib", sys.argv[4])):
    if not raw and name == "gpu-memory-gib":
        continue
    number = float(raw)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + " must be finite and positive")
with open(sys.argv[1], encoding="utf-8") as handle:
    schema = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    config = resolve_config(json.load(handle))
if schema["dataset"] != "cebab":
    raise ValueError("CEBaB only")
if "nano_risk" in config["evaluation"]["methods"]:
    raise ValueError("use explicit Nano workflow")
if ({"risk", "value"} - {config["learning"]["objective"]}) & set(config["evaluation"]["methods"]):
    raise ValueError("controller objective/method mismatch")
' "$prepared/schema.json" "$config" "$gpu_memory_gib" "$gpu_reserve_gib"

prefix=(env)
if [[ -n "$gpu" ]]; then prefix=(env "CUDA_VISIBLE_DEVICES=$gpu"); fi
python_args=(-m cbmjev)
if [[ -n "$gpu_memory_gib" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || fail 'nvidia-smi is required to resolve the selected GPU UUID'
  gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader) || fail 'could not resolve GPU UUID'
  [[ "$gpu_uuid" =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]] \
    || fail 'expected exactly one valid physical GPU UUID from nvidia-smi'
  prefix=(env "CUDA_VISIBLE_DEVICES=$gpu_uuid")
  python_args=("$CBMJEV_PROJECT_DIR/tools/run_gpu_limited.py" --memory-gib "$gpu_memory_gib" \
    --reserve-gib "$gpu_reserve_gib" -- -m cbmjev)
fi
child_pid=''
interrupt_child() {
  trap - INT TERM
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  printf 'Interrupted; partial output retained at %s.\n' "$out" >&2
  exit 130
}
trap interrupt_child INT TERM
run() {
  printf '+'
  printf ' %q' "${prefix[@]}" "$python_exe" "${python_args[@]}" "$@"
  printf '\n'
  if [[ "$dry_run" -eq 0 ]]; then
    "${prefix[@]}" "$python_exe" "${python_args[@]}" "$@" &
    child_pid=$!
    wait "$child_pid"
    child_pid=''
  fi
}
if [[ "$dry_run" -eq 0 ]]; then
  [[ -d "$(dirname "$out")" ]] || fail 'output parent must already exist'
  mkdir "$out"
fi
run doctor --check-device "$device" --out "$out/doctor.json"
responder_args=(train-responder --prepared "$prepared" --kind "$responder_kind" --out "$out/responder"
  --device "$device" --seed "$seed" --epochs "$epochs")
if [[ "$responder_kind" == hf_text ]]; then
  responder_args+=(--backbone "$backbone" --max-length "$max_length")
  [[ -z "$revision" ]] || responder_args+=(--revision "$revision")
  [[ "$allow_download" -eq 0 ]] || responder_args+=(--allow-download)
  [[ "$freeze_backbone" -eq 0 ]] || responder_args+=(--freeze-backbone)
fi
run "${responder_args[@]}"
run cache --prepared "$prepared" --responder "$out/responder" --out "$out/cache" --device "$device"
run audit-responses --prepared "$prepared" --cache "$out/cache" --out "$out/semantic_audit" --split validation
run train --cache "$out/cache" --config "$config" --out "$out/models" --seed "$seed" --device "$device"
run evaluate --models "$out/models" --cache "$out/cache" --out "$out/validation_replay" --device "$device"
summary_runs=("$out/validation_replay")
if [[ "$live" -eq 1 ]]; then
  run evaluate --models "$out/models" --cache "$out/cache" --prepared "$prepared" \
    --responder "$out/responder" --out "$out/validation_live" --device "$device" --warmup 3
  summary_runs+=("$out/validation_live")
fi
run summarize --runs "${summary_runs[@]}" --out "$out/tables"
printf 'Chain %s. Data split unchanged; seed=%s; separate responder fitted; test not evaluated.\n' \
  "$([[ "$dry_run" -eq 1 ]] && printf planned || printf finished)" "$seed"
