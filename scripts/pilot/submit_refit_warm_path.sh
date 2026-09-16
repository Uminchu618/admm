#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
cv_output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv}"
refit_output_dir="${PILOT_REFIT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refit}"
warm_dir="${PILOT_REFIT_WARM_DIR:-${refit_output_dir}_warm_paths}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/mcp_config.toml}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
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

mkdir -p "$repo_root/logs/pilot_refit_warm_path"
cd "$repo_root"
qsub -t "1-${#train_files[@]}:1" -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_EVAL_DIR=$eval_dir,PILOT_OUTPUT_DIR=$cv_output_dir,PILOT_REFIT_OUTPUT_DIR=$refit_output_dir,PILOT_REFIT_WARM_DIR=$warm_dir,PILOT_CONFIG_TEMPLATE=$config_template,PILOT_LAMBDA_GRID=$lambda_grid" qsub_pilot_refit_warm_path.sh
