#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"
cv_output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv/$run_name}"
refit_output_dir="${PILOT_REFIT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refit/$run_name}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/diagnostic_config.toml}"
uv_bin="${UV_BIN:-$(command -v uv)}"

shopt -s nullglob
train_files=("$train_dir"/*.csv)
shopt -u nullglob
if [ "${#train_files[@]}" -eq 0 ]; then
	echo "No training CSVs found in $train_dir" >&2
	exit 1
fi
for train_path in "${train_files[@]}"; do
	data_name="$(basename "$train_path" .csv)"
	if [ ! -f "$eval_dir/$data_name.csv" ] || [ ! -f "$cv_output_dir/$data_name/selected_lambda.json" ]; then
		echo "Missing eval data or CV selection for $data_name" >&2
		exit 1
	fi
done

mkdir -p "$repo_root/logs/pilot_refit"
cd "$repo_root"
qsub -t "1-${#train_files[@]}:1" -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_EVAL_DIR=$eval_dir,PILOT_OUTPUT_DIR=$cv_output_dir,PILOT_REFIT_OUTPUT_DIR=$refit_output_dir,PILOT_CONFIG_TEMPLATE=$config_template,PILOT_RUN_NAME=$run_name" qsub_pilot_refit.sh
