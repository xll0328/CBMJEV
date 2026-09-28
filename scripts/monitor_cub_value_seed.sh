#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: monitor_cub_value_seed.sh <seed> [gpu] [full|canonical_fixed_only] }"
gpu="${2:-0}"
mode="${3:-full}"

if [[ ! "$seed" =~ ^[0-9]+$ ]] || [[ "$gpu" != "0" && "$gpu" != "1" ]] || \
   [[ "$mode" != "full" && "$mode" != "canonical_fixed_only" ]]; then
  echo "[monitor] invalid seed/GPU/mode; only physical GPU 0/1 authorized" >&2
  exit 3
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

prepared="data/prepared/cub_official_seed17"
planned="runs/cub_crossfit_plan_v1"
raw_root="data/raw/cub_200_2011_official/extracted_v1/CUB_200_2011"
outer0="runs/cub_outer0_value_seed${seed}_v1"
outer1="runs/cub_outer1_value_seed${seed}_v1"
outer2="runs/cub_outer2_value_seed${seed}_v1"
merged="runs/cub_merged_value_seed${seed}_v1"
static_order="runs/cub_static_value_seed${seed}_v1"
final_responder_nested="runs/cub_final_balanced_seed${seed}_v1/responder"
final_responder_flat="runs/cub_final_balanced_seed${seed}_v1"
final_responder_current="runs/cub_final_balanced_seed${seed}_current_v1/responder"
final_responder_current_pid="logs/cub_final_balanced_seed${seed}_current_v1.pid"
final_cache="runs/cub_final_balanced_seed${seed}_validation_cache_v1"
eval_out="runs/cub_eval_value_budget_grid_seed${seed}_v1"

echo "[monitor] seed=${seed} gpu=${gpu} mode=${mode} cwd=$PWD"

if [[ "$mode" == "canonical_fixed_only" ]]; then
  canonical_out="runs/cub_eval_canonical_fixed_seed${seed}_v1"
  if [[ -e "$canonical_out" ]]; then
    echo "[monitor] canonical output exists; never overwrite: $canonical_out" >&2
    exit 3
  fi
  for source in "$prepared/audit.json" "$planned/plan.json" "$merged/receipt.json" \
                "$final_cache/manifest.json" "$eval_out/metrics.json"; do
    if [[ ! -f "$source" ]]; then
      echo "[monitor] canonical replay missing source: $source" >&2
      exit 2
    fi
  done
  if [[ ! -x .venv/bin/python ]] || ! command -v sha256sum >/dev/null; then
    echo "[monitor] Python environment or SHA-256 utility unavailable" >&2
    exit 3
  fi
  expected_responder_sha="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_binding"]["responder_receipt_sha256"])' "$eval_out/metrics.json")"
  final_responder=""
  for candidate in "$final_responder_current" "$final_responder_nested" "$final_responder_flat"; do
    if [[ -f "$candidate/receipt.json" ]] && \
       [[ "$(sha256sum "$candidate/receipt.json" | cut -d ' ' -f 1)" == "$expected_responder_sha" ]]; then
      final_responder="$candidate"
      break
    fi
  done
  if [[ -z "$final_responder" ]]; then
    echo "[monitor] no responder matches historical receipt SHA-256" >&2
    exit 3
  fi
  if ! command -v nvidia-smi >/dev/null; then
    echo "[monitor] nvidia-smi unavailable; cannot check GPU headroom" >&2
    exit 3
  fi
  free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
    echo "[monitor] insufficient verified free GPU memory on physical GPU $gpu: ${free_mib} MiB" >&2
    exit 3
  fi
  available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < 16777216 )); then
    echo "[monitor] insufficient verified host memory: ${available_kib} KiB" >&2
    exit 3
  fi
  echo "[monitor] canonical fixed replay -> $canonical_out; responder=$final_responder; GPU${gpu} free=${free_mib}MiB"
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m scripts.evaluate_crossfit_budget_grid \
    --prepared "$prepared" --planned "$planned" --merged "$merged" \
    --responder "$final_responder" --cache "$final_cache" \
    --expected-grid "$eval_out/metrics.json" --out "$canonical_out" \
    --budgets 0 1 2 4 8 16 28 --methods fixed --device cuda:0
  exit 0
fi

for d in "$outer0" "$outer1" "$outer2"; do
  if [[ ! -f "$d/receipt.json" ]]; then
    echo "[monitor] waiting: missing $d/receipt.json"
    exit 2
  fi
done

if [[ ! -d "$merged" ]]; then
  echo "[monitor] merging outer folds -> $merged"
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m cbmjev.cli merge-crossfit-outer \
    --prepared "$prepared" \
    --planned "$planned" \
    --outer-dirs "$outer0" "$outer1" "$outer2" \
    --out "$merged" \
    --controller-batch-size 4096
else
  echo "[monitor] merged exists: $merged"
fi

if [[ ! -d "$static_order" ]]; then
  echo "[monitor] fitting static order -> $static_order"
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m cbmjev.cli fit-crossfit-static \
    --prepared "$prepared" \
    --planned "$planned" \
    --outer-dirs "$outer0" "$outer1" "$outer2" \
    --out "$static_order" \
    --device "cuda:0" \
    --batch-size 256
else
  echo "[monitor] static order exists: $static_order"
fi

current_final_pid=""
if [[ -f "$final_responder_current_pid" ]]; then
  current_final_pid="$(tr -d '[:space:]' < "$final_responder_current_pid" 2>/dev/null || true)"
fi

if [[ -f "$final_responder_current/receipt.json" ]]; then
  final_responder="$final_responder_current"
elif [[ -n "$current_final_pid" ]] && kill -0 "$current_final_pid" 2>/dev/null; then
  echo "[monitor] waiting: current final responder pid $current_final_pid"
  exit 2
elif [[ -d "$(dirname "$final_responder_current")" ]]; then
  echo "[monitor] waiting: current final responder in progress at $final_responder_current"
  exit 2
elif [[ -f "$final_responder_nested/receipt.json" ]]; then
  final_responder="$final_responder_nested"
elif [[ -f "$final_responder_flat/receipt.json" ]]; then
  final_responder="$final_responder_flat"
else
  echo "[monitor] final responder missing: expected $final_responder_nested/receipt.json or $final_responder_flat/receipt.json"
  exit 3
fi
echo "[monitor] final responder: $final_responder"

if [[ ! -d "$final_cache" ]]; then
  echo "[monitor] caching final validation responses -> $final_cache"
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m cbmjev.cli cache-crossfit \
    --prepared "$prepared" \
    --planned "$planned" \
    --responder "$final_responder" \
    --out "$final_cache" \
    --raw-root "$raw_root" \
    --device "cuda:0" \
    --split validation
else
  echo "[monitor] final validation cache exists: $final_cache"
fi

if [[ ! -d "$eval_out" ]]; then
  echo "[monitor] evaluating value budget grid -> $eval_out"
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m scripts.evaluate_crossfit_budget_grid \
    --prepared "$prepared" \
    --planned "$planned" \
    --merged "$merged" \
    --responder "$final_responder" \
    --cache "$final_cache" \
    --static-order-dir "$static_order" \
    --out "$eval_out" \
    --budgets 0 1 2 4 8 16 28 \
    --methods stop fixed random static value value_singleton static_value all \
    --device "cuda:0"
else
  echo "[monitor] evaluation exists: $eval_out"
fi

echo "[monitor] complete seed=${seed}"
