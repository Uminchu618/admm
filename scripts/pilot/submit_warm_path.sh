#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_penalty_warm_path}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/mcp_config.toml}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
n_folds="${PILOT_N_FOLDS:-5}"
split_seed="${PILOT_SPLIT_SEED:-1234}"
expected_datasets="${PILOT_EXPECTED_DATASETS:-100}"
uv_bin="${UV_BIN:-$(command -v uv)}"
initial_result_base="${PILOT_INITIAL_RESULT_BASE:-}"

shopt -s nullglob
train_files=("$train_dir"/*.csv)
shopt -u nullglob
if [ "${#train_files[@]}" -ne "$expected_datasets" ]; then
	echo "Expected $expected_datasets train CSVs; found ${#train_files[@]}." >&2
	exit 1
fi
if [ "$(jq '.lambda_values | length' "$lambda_grid")" -eq 0 ]; then
	echo "Empty lambda grid: $lambda_grid" >&2
	exit 1
fi

total_tasks=$((${#train_files[@]} * n_folds))
mkdir -p "$repo_root/logs/pilot_warm_path"
cd "$repo_root"
qsub -t "1-${total_tasks}:1" -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_OUTPUT_DIR=$output_dir,PILOT_CONFIG_TEMPLATE=$config_template,PILOT_LAMBDA_GRID=$lambda_grid,PILOT_N_FOLDS=$n_folds,PILOT_SPLIT_SEED=$split_seed,PILOT_INITIAL_RESULT_BASE=$initial_result_base" qsub_pilot_warm_path.sh
