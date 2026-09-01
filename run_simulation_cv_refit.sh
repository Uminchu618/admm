#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
data_dir="${DATA_DIR:-$repo_root/data/extended_aft_step}"
eval_data_dir="${EVAL_DATA_DIR:-$repo_root/data/extended_aft_step_eval}"
cv_output_dir="${CV_OUTPUT_DIR:-$repo_root/outputs/simulation_cv}"
output_base_dir="${OUTPUT_BASE_DIR:-$repo_root/outputs/simulation_cv_refit}"
config_template="${CONFIG_TEMPLATE:-$repo_root/config.toml}"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"

shopt -s nullglob
data_files=("$data_dir"/*.csv)
shopt -u nullglob
if [ "${#data_files[@]}" -eq 0 ]; then
	echo "No CSV files found in $data_dir" >&2
	exit 1
fi

total_tasks="${#data_files[@]}"
task_id="${SGE_TASK_ID:-${1:-1}}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$total_tasks" ]; then
	echo "Task ID out of range: $task_id (1..$total_tasks)" >&2
	exit 1
fi

selected_data="${data_files[$((task_id - 1))]}"
data_name="$(basename "$selected_data" .csv)"
selected_eval="$eval_data_dir/$(basename "$selected_data")"
selection_file="$cv_output_dir/$data_name/selected_lambda.json"
for required in "$selected_data" "$selected_eval" "$selection_file" "$config_template"; do
	if [ ! -f "$required" ]; then
		echo "Required file not found: $required" >&2
		exit 1
	fi
done

selected_lambda="$("$uv_bin" run python - "$selection_file" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = float(payload["selected_lambda"])
if payload.get("selection_method") != "five_fold_cv_mean_c_td":
    raise SystemExit("Unexpected selection method")
if not math.isfinite(value) or value < 0:
    raise SystemExit("Invalid selected lambda")
print(value)
PY
)"

output_dir="$(printf '%s/%s/lambda_%.15g' "$output_base_dir" "$data_name" "$selected_lambda")"
output_json="$output_dir/result.json"
if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$output_json" ]; then
	echo "Skip existing result: $output_json"
	exit 0
fi

mkdir -p "$output_dir"
cp "$selection_file" "$output_dir/selected_lambda.json"
temp_config="$output_dir/config.toml"
cp "$config_template" "$temp_config"
sed -i.bak "s/^lambda_fuse = .*/lambda_fuse = $selected_lambda/" "$temp_config"
rm -f "$temp_config.bak"

echo "Simulation CV refit $task_id / $total_tasks: $data_name lambda=$selected_lambda"
cd "$repo_root"
"$uv_bin" run main.py \
	--config "$temp_config" \
	--data "$selected_data" \
	--eval-data "$selected_eval" \
	--output "$output_json"

echo "Saved independent-evaluation result to: $output_json"
