#!/usr/bin/env bash
set -euo pipefail

seed_key="${1:-60_61_62}"
tag="${2:-cub_value_policy}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

aggregate_dir="results/main/cub_value_budget_grid_${seed_key}"

required=(
  "$aggregate_dir/frontier_long.csv"
  "$aggregate_dir/frontier_summary.json"
  "$aggregate_dir/frontier_table.tex"
  "$aggregate_dir/frontier_accuracy.pdf"
  "$aggregate_dir/frontier_macro_f1.pdf"
  "$aggregate_dir/claim_analysis_accuracy.json"
  "$aggregate_dir/claim_analysis_accuracy.md"
  "$aggregate_dir/claim_analysis_macro_f1.json"
  "$aggregate_dir/claim_analysis_macro_f1.md"
  "$aggregate_dir/paired_seed_deltas_accuracy.json"
  "$aggregate_dir/paired_seed_deltas_accuracy.md"
  "$aggregate_dir/paired_seed_deltas_macro_f1.json"
  "$aggregate_dir/paired_seed_deltas_macro_f1.md"
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[paper-stage] waiting: missing $path"
    exit 2
  fi
done

manifest="paper/cvpr2027/generated/${tag}_manifest.json"
update_json="results/main/cub_value_policy_paper_update_${seed_key}.json"
update_md="results/main/CUB_VALUE_POLICY_PAPER_UPDATE_${seed_key}.md"

if [[ -f "$manifest" ]]; then
  echo "[paper-stage] staged exists: $manifest"
else
  "$python_bin" scripts/figures/stage_cub_value_paper_assets.py \
    --aggregate-dir "$aggregate_dir" \
    --paper-dir paper/cvpr2027 \
    --tag "$tag"
fi

"$python_bin" scripts/prepare_cub_value_paper_update.py \
  --aggregate-dir "$aggregate_dir" \
  --paper-dir paper/cvpr2027 \
  --tag "$tag" \
  --out-json "$update_json" \
  --out-md "$update_md" \
  --apply-evidence-state

if [[ -f scripts/report_delivery_gates.py ]]; then
  "$python_bin" scripts/report_delivery_gates.py \
    --out-json results/main/delivery_gates_20260923.json \
    --out-md results/main/DELIVERY_GATES_20260923.md
fi

echo "[paper-stage] complete $manifest $update_json"
