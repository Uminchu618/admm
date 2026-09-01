#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"
output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv/$run_name}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/diagnostic_config.toml}"
uv_bin="${UV_BIN:-$(command -v uv)}"

shopt -s nullglob
train_files=("$train_dir"/*.csv)
eval_files=("$eval_dir"/*.csv)
shopt -u nullglob
n_lambda="$(jq '.lambda_values | length' "$lambda_grid")"
n_folds="${PILOT_N_FOLDS:-5}"
split_seed="${PILOT_SPLIT_SEED:-1234}"
expected_tasks=$((${#train_files[@]} * n_lambda * n_folds))

if [ "${#train_files[@]}" -ne 100 ] || [ "${#eval_files[@]}" -ne 100 ]; then
	echo "Expected 100 train and 100 eval CSVs; found ${#train_files[@]} and ${#eval_files[@]}." >&2
	exit 1
fi
if [ "$n_lambda" -ne 9 ]; then
	echo "Expected 9 validated lambda values; found $n_lambda in $lambda_grid." >&2
	exit 1
fi

mkdir -p "$repo_root/logs/pilot"
cd "$repo_root"
qsub -t "1-${expected_tasks}:1" -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_EVAL_DIR=$eval_dir,PILOT_OUTPUT_DIR=$output_dir,PILOT_CONFIG_TEMPLATE=$config_template,PILOT_LAMBDA_GRID=$lambda_grid,PILOT_RUN_NAME=$run_name,PILOT_N_FOLDS=$n_folds,PILOT_SPLIT_SEED=$split_seed" qsub_pilot.sh
