#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
lambda_grid="${PILOT_DIAGNOSTIC_LAMBDA_GRID:-$repo_root/generation/pilot/diagnostic_lambda_grid.json}"
config_template="${PILOT_DIAGNOSTIC_CONFIG:-$repo_root/generation/pilot/diagnostic_config.toml}"
run_name="${PILOT_DIAGNOSTIC_RUN:-adaptive_rho_normalized_newton5}"
output_base="${PILOT_DIAGNOSTIC_OUTPUT_DIR:-$repo_root/outputs/pilot_diagnostic/$run_name}"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"
seed_start="${PILOT_DIAGNOSTIC_SEED_START:-42}"
seed_end="${PILOT_DIAGNOSTIC_SEED_END:-44}"

data_files=()
for scenario in oracle fine_grid; do
	for ((seed = seed_start; seed <= seed_end; seed++)); do
		data_files+=("$train_dir/${scenario}_seed_${seed}.csv")
	done
done

lambda_values=()
while IFS= read -r lambda_value; do
	lambda_values+=("$lambda_value")
done < <(jq -r '.lambda_values[]' "$lambda_grid")

n_data="${#data_files[@]}"
n_lambda="${#lambda_values[@]}"
total_patterns=$((n_data * n_lambda))
task_id="${SGE_TASK_ID:-${1:-1}}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$total_patterns" ]; then
	echo "Task ID out of range: $task_id (1..$total_patterns)" >&2
	exit 1
fi

task_idx=$((task_id - 1))
data_idx=$((task_idx / n_lambda))
lambda_idx=$((task_idx % n_lambda))
selected_data="${data_files[$data_idx]}"
selected_eval="$eval_dir/$(basename "$selected_data")"
selected_lambda="${lambda_values[$lambda_idx]}"

for required in "$selected_data" "$selected_eval" "$lambda_grid" "$config_template"; do
	if [ ! -f "$required" ]; then
		echo "Required file not found: $required" >&2
		exit 1
	fi
done

data_name="$(basename "$selected_data" .csv)"
output_dir="$output_base/$data_name/lambda_${selected_lambda}"
mkdir -p "$output_dir"
temp_config="$output_dir/config.toml"
cp "$config_template" "$temp_config"
sed -i.bak "s/^lambda_fuse = .*/lambda_fuse = $selected_lambda/" "$temp_config"

if [ -n "${DIAGNOSTIC_RHO:-}" ]; then
	sed -i.bak "s/^rho = .*/rho = $DIAGNOSTIC_RHO/" "$temp_config"
fi
if [ -n "${DIAGNOSTIC_ADAPTIVE_RHO:-}" ]; then
	sed -i.bak "s/^adaptive_rho = .*/adaptive_rho = $DIAGNOSTIC_ADAPTIVE_RHO/" "$temp_config"
fi
if [ -n "${DIAGNOSTIC_NEWTON_STEPS:-}" ]; then
	sed -i.bak "s/^newton_steps_per_admm = .*/newton_steps_per_admm = $DIAGNOSTIC_NEWTON_STEPS/" "$temp_config"
fi
rm -f "$temp_config.bak"

output_json="$output_dir/result.json"
if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$output_json" ]; then
	echo "Skip existing result: $output_json"
	exit 0
fi

echo "Diagnostic task $task_id / $total_patterns: $data_name lambda=$selected_lambda run=$run_name"
cd "$repo_root"
"$uv_bin" run main.py \
	--config "$temp_config" \
	--data "$selected_data" \
	--eval-data "$selected_eval" \
	--output "$output_json"
