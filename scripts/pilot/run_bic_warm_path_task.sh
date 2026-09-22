#!/bin/bash
set -euo pipefail

# 1 task = 1 dataset。全学習データで lambda を降順に解き、全候補を保存する。
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
data_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
output_base_dir="${PILOT_BIC_OUTPUT_DIR:-$repo_root/outputs/pilot_bic_selection/mcp}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/mcp_config.toml}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
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

task_id="${SGE_TASK_ID:-${1:-1}}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "${#data_files[@]}" ]; then
	echo "Task ID out of range: $task_id (1..${#data_files[@]})" >&2
	exit 1
fi

selected_data="${data_files[$((task_id - 1))]}"
data_name="$(basename "$selected_data" .csv)"
selected_eval="$eval_dir/$data_name.csv"
if [ ! -f "$selected_eval" ]; then
	echo "Matching evaluation CSV not found: $selected_eval" >&2
	exit 1
fi

previous_result=""
for lambda_value in "${lambda_values[@]}"; do
	output_dir="$(printf '%s/%s/lambda_%.15g' "$output_base_dir" "$data_name" "$lambda_value")"
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
		initial_candidate="$(printf '%s/%s/lambda_%.15g/result.json' "$initial_result_base" "$data_name" "$lambda_value")"
		if [ ! -f "$initial_candidate" ]; then
			echo "Initial result not found: $initial_candidate" >&2
			exit 1
		fi
		previous_result="$initial_candidate"
	fi

	command=(
		"$uv_bin" run python main.py
		--config "$temp_config"
		--data "$selected_data"
		--eval-data "$selected_eval"
		--output "$output_json"
	)
	if [ -n "$previous_result" ]; then
		command+=(--init-result "$previous_result")
	fi
	"${command[@]}"
	previous_result="$output_json"
done

echo "Completed full-data BIC lambda path: $data_name"
