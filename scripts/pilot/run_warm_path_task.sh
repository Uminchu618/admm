#!/bin/bash
set -euo pipefail

# 1 task = 1 dataset × 1 fold。lambda を大きい順に逐次実行し、直前解を初期値にする。
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
data_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
output_base_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_penalty_warm_path}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/mcp_config.toml}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
n_folds="${PILOT_N_FOLDS:-5}"
split_seed="${PILOT_SPLIT_SEED:-1234}"
uv_bin="${UV_BIN:-uv}"
initial_result_base="${PILOT_INITIAL_RESULT_BASE:-}"

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
done < <(jq -r '.lambda_values | sort | reverse[]' "$lambda_grid")
if [ "${#lambda_values[@]}" -eq 0 ]; then
	echo "No lambda values found in $lambda_grid" >&2
	exit 1
fi

total_tasks=$((${#data_files[@]} * n_folds))
task_id="${SGE_TASK_ID:-${1:-1}}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$total_tasks" ]; then
	echo "Task ID out of range: $task_id (1..$total_tasks)" >&2
	exit 1
fi

task_idx=$((task_id - 1))
data_idx=$((task_idx / n_folds))
fold_idx=$((task_idx % n_folds))
selected_data="${data_files[$data_idx]}"
data_name="$(basename "$selected_data" .csv)"
fold_work="$output_base_dir/_fold_data/$data_name/fold_$(printf '%02d' "$fold_idx")"

mkdir -p "$fold_work"
cd "$repo_root"
"$uv_bin" run python scripts/simulation_cv/prepare_fold.py \
	--data "$selected_data" \
	--output-dir "$fold_work" \
	--fold "$fold_idx" \
	--n-folds "$n_folds" \
	--random-state "$split_seed"

previous_result=""
for lambda_value in "${lambda_values[@]}"; do
	output_dir="$(printf '%s/%s/lambda_%.15g/fold_%02d' "$output_base_dir" "$data_name" "$lambda_value" "$fold_idx")"
	output_json="$output_dir/result.json"
	if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$output_json" ]; then
		previous_result="$output_json"
		continue
	fi

	mkdir -p "$output_dir"
	temp_config="$output_dir/config.toml"
	cp "$config_template" "$temp_config"
	sed -i.bak "s/^lambda_fuse = .*/lambda_fuse = $lambda_value/" "$temp_config"
	rm -f "$temp_config.bak"

	if [ -z "$previous_result" ] && [ -n "$initial_result_base" ]; then
		initial_candidate="$(printf '%s/%s/lambda_%.15g/fold_%02d/result.json' "$initial_result_base" "$data_name" "$lambda_value" "$fold_idx")"
		if [ ! -f "$initial_candidate" ]; then
			echo "Initial result not found: $initial_candidate" >&2
			exit 1
		fi
		previous_result="$initial_candidate"
	fi

	command=(
		"$uv_bin" run python main.py
		--config "$temp_config"
		--data "$fold_work/data/train.csv"
		--eval-data "$fold_work/data/test.csv"
		--output "$output_json"
	)
	if [ -n "$previous_result" ]; then
		command+=(--init-result "$previous_result")
	fi
	"${command[@]}"
	previous_result="$output_json"
done

echo "Completed descending-lambda warm path: $data_name fold=$fold_idx"
