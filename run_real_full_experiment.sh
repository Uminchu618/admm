#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"

datasets_raw="${DATASETS:-support2,framingham}"
config_template="${CONFIG_PATH:-$repo_root/config.toml}"
lambda_grid_file="${LAMBDA_GRID:-$repo_root/lambda_grid.json}"
experiment_name="${EXPERIMENT_NAME:-full_data}"
output_base_dir="${OUTPUT_BASE_DIR:-$repo_root/outputs/real_full}"

IFS=',' read -r -a datasets <<< "$datasets_raw"
if [ "${#datasets[@]}" -eq 0 ]; then
	echo "No DATASETS specified" >&2
	exit 1
fi

if [ ! -f "$lambda_grid_file" ]; then
	echo "Lambda grid file not found: $lambda_grid_file" >&2
	exit 1
fi

mapfile -t lambda_values < <("$uv_bin" run python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$lambda_grid_file").read_text(encoding="utf-8"))
for value in payload["lambda_values"]:
    print(value)
PY
)

if [ "${#lambda_values[@]}" -eq 0 ]; then
	echo "No lambda values found in $lambda_grid_file" >&2
	exit 1
fi

if [ -z "${SGE_TASK_ID:-}" ]; then
	if [ $# -ge 1 ]; then
		SGE_TASK_ID="$1"
	else
		SGE_TASK_ID=1
	fi
fi

n_lambda="${#lambda_values[@]}"
n_datasets="${#datasets[@]}"
total_tasks=$((n_lambda * n_datasets))

if [ "$SGE_TASK_ID" -lt 1 ] || [ "$SGE_TASK_ID" -gt "$total_tasks" ]; then
	echo "SGE_TASK_ID out of range: $SGE_TASK_ID (1..$total_tasks)" >&2
	exit 1
fi

task_idx=$((SGE_TASK_ID - 1))
dataset_idx=$((task_idx / n_lambda))
lambda_idx=$((task_idx % n_lambda))
dataset="${datasets[$dataset_idx]}"
selected_lambda="${lambda_values[$lambda_idx]}"

case "$dataset" in
	support2)
		input_csv="${SUPPORT2_INPUT:-$repo_root/data/real/support/support2.csv}"
		;;
	framingham)
		input_csv="${FRAMINGHAM_INPUT:-$repo_root/data/real/framingham/framingham.csv}"
		;;
	*)
		echo "Unsupported DATASET in DATASETS: $dataset" >&2
		exit 1
		;;
esac

lambda_dir="$(printf 'lambda_%.15g' "$selected_lambda")"
output_dir="$output_base_dir/$dataset/$experiment_name/$lambda_dir"

echo "=== Full real-data task $SGE_TASK_ID / $total_tasks ==="
echo "Dataset: $dataset"
echo "Experiment: $experiment_name"
echo "Lambda: $selected_lambda"
echo "Output: $output_dir"

cd "$repo_root"

"$uv_bin" run scripts/real_cv/prepare_full_data.py \
	--dataset "$dataset" \
	--input "$input_csv" \
	--config "$config_template" \
	--lambda-fuse "$selected_lambda" \
	--output-dir "$output_dir"

"$uv_bin" run main.py \
	--config "$output_dir/config.json" \
	--data "$output_dir/data/all.csv" \
	--output "$output_dir/result.json"

echo "Saved result to: $output_dir/result.json"
