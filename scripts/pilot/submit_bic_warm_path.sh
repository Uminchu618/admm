#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
output_dir="${PILOT_BIC_OUTPUT_DIR:-$repo_root/outputs/pilot_bic_selection/mcp}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/mcp_config.toml}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
expected_datasets="${PILOT_EXPECTED_DATASETS:-100}"
uv_bin="${UV_BIN:-$(command -v uv)}"
initial_result_base="${PILOT_INITIAL_RESULT_BASE:-}"
skip_existing="${SKIP_EXISTING:-0}"

shopt -s nullglob
train_files=("$train_dir"/*.csv)
shopt -u nullglob
if [ "${#train_files[@]}" -ne "$expected_datasets" ]; then
	echo "Expected $expected_datasets train CSVs; found ${#train_files[@]}." >&2
	exit 1
fi
for train_path in "${train_files[@]}"; do
	data_name="$(basename "$train_path" .csv)"
	if [ ! -f "$eval_dir/$data_name.csv" ]; then
		echo "Missing evaluation data for $data_name" >&2
		exit 1
	fi
done
if [ "$(jq '.lambda_values | length' "$lambda_grid")" -eq 0 ]; then
	echo "Empty lambda grid: $lambda_grid" >&2
	exit 1
fi

mkdir -p "$repo_root/logs/pilot_bic_warm_path"
cd "$repo_root"
qsub -t "1-${#train_files[@]}:1" -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_EVAL_DIR=$eval_dir,PILOT_BIC_OUTPUT_DIR=$output_dir,PILOT_CONFIG_TEMPLATE=$config_template,PILOT_LAMBDA_GRID=$lambda_grid,PILOT_INITIAL_RESULT_BASE=$initial_result_base,SKIP_EXISTING=$skip_existing" qsub_pilot_bic_warm_path.sh
