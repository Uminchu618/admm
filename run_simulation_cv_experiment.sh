#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
data_dir="${DATA_DIR:-$repo_root/data/extended_aft_step}"
output_base_dir="${OUTPUT_BASE_DIR:-$repo_root/outputs/simulation_cv}"
config_template="${CONFIG_TEMPLATE:-$repo_root/config.toml}"
lambda_grid_file="${LAMBDA_GRID_FILE:-$repo_root/lambda_grid.json}"
n_folds="${N_FOLDS:-5}"
split_seed="${SPLIT_SEED:-1234}"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"

shopt -s nullglob
data_files=("$data_dir"/*.csv)
shopt -u nullglob
if [ "${#data_files[@]}" -eq 0 ]; then
	echo "No CSV files found in $data_dir" >&2
	exit 1
fi

lambda_values=()
while IFS= read -r lambda_value; do
	lambda_values+=("$lambda_value")
done < <(jq -r '.lambda_values[]' "$lambda_grid_file")
if [ "${#lambda_values[@]}" -eq 0 ]; then
	echo "No lambda values found in $lambda_grid_file" >&2
	exit 1
fi

n_data="${#data_files[@]}"
n_lambda="${#lambda_values[@]}"
total_patterns=$((n_data * n_lambda * n_folds))
task_id="${SGE_TASK_ID:-${1:-1}}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$total_patterns" ]; then
	echo "Task ID out of range: $task_id (1..$total_patterns)" >&2
	exit 1
fi

task_idx=$((task_id - 1))
data_idx=$((task_idx / (n_lambda * n_folds)))
remainder=$((task_idx % (n_lambda * n_folds)))
lambda_idx=$((remainder / n_folds))
fold_idx=$((remainder % n_folds))
selected_data="${data_files[$data_idx]}"
selected_lambda="${lambda_values[$lambda_idx]}"
data_name="$(basename "$selected_data" .csv)"
output_dir="$(printf '%s/%s/lambda_%.15g/fold_%02d' "$output_base_dir" "$data_name" "$selected_lambda" "$fold_idx")"
output_json="$output_dir/result.json"

if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$output_json" ]; then
	echo "Skip existing result: $output_json"
	exit 0
fi

mkdir -p "$output_dir"
temp_config="$output_dir/config.toml"
cp "$config_template" "$temp_config"
sed -i.bak "s/^lambda_fuse = .*/lambda_fuse = $selected_lambda/" "$temp_config"
rm -f "$temp_config.bak"

echo "Simulation CV task $task_id / $total_patterns"
echo "Data: $selected_data"
echo "Lambda: $selected_lambda"
echo "Fold: $fold_idx / $n_folds"

cd "$repo_root"
"$uv_bin" run scripts/simulation_cv/prepare_fold.py \
	--data "$selected_data" \
	--output-dir "$output_dir" \
	--fold "$fold_idx" \
	--n-folds "$n_folds" \
	--random-state "$split_seed"

"$uv_bin" run main.py \
	--config "$temp_config" \
	--data "$output_dir/data/train.csv" \
	--eval-data "$output_dir/data/test.csv" \
	--output "$output_json"

echo "Saved result to: $output_json"
