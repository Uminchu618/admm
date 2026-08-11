#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/config.toml}"
uv_bin="${UV_BIN:-$(command -v uv)}"

shopt -s nullglob
train_files=("$train_dir"/*.csv)
eval_files=("$eval_dir"/*.csv)
shopt -u nullglob
n_lambda="$(jq '.lambda_values | length' "$lambda_grid")"
expected_tasks=$((${#train_files[@]} * n_lambda))

if [ "${#train_files[@]}" -ne 100 ] || [ "${#eval_files[@]}" -ne 100 ]; then
	echo "Expected 100 train and 100 eval CSVs; found ${#train_files[@]} and ${#eval_files[@]}." >&2
	exit 1
fi
if [ "$expected_tasks" -ne 1200 ]; then
	echo "Expected 1200 tasks; computed $expected_tasks." >&2
	exit 1
fi

mkdir -p "$repo_root/logs/pilot"
cd "$repo_root"
qsub -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_EVAL_DIR=$eval_dir,PILOT_OUTPUT_DIR=$output_dir,PILOT_CONFIG_TEMPLATE=$config_template,PILOT_LAMBDA_GRID=$lambda_grid" qsub_pilot.sh
