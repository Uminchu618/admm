#!/bin/bash
set -euo pipefail

# CV 選択 lambda まで全学習データ上で降順 lambda path を解き、最後だけ本評価へ保存する。
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
data_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
cv_output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv}"
output_base_dir="${PILOT_REFIT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refit}"
warm_base_dir="${PILOT_REFIT_WARM_DIR:-${output_base_dir}_warm_paths}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/mcp_config.toml}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
uv_bin="${UV_BIN:-uv}"

shopt -s nullglob
data_files=("$data_dir"/*.csv)
shopt -u nullglob
if [ "${#data_files[@]}" -eq 0 ]; then
	echo "No CSV files found in $data_dir" >&2
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
selection_file="$cv_output_dir/$data_name/selected_lambda.json"
for required in "$selected_data" "$selected_eval" "$selection_file" "$config_template" "$lambda_grid"; do
	if [ ! -f "$required" ]; then
		echo "Required file not found: $required" >&2
		exit 1
	fi
done

selected_lambda="$($uv_bin run python - "$selection_file" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = float(payload["selected_lambda"])
if payload.get("selection_method") != "five_fold_cv_mean_c_td":
    raise SystemExit("Unexpected selection method")
if not math.isfinite(value) or value < 0.0:
    raise SystemExit("Invalid selected lambda")
print(value)
PY
)"

lambda_values=()
while IFS= read -r lambda_value; do
	lambda_values+=("$lambda_value")
done < <("$uv_bin" run python - "$lambda_grid" "$selected_lambda" <<'PY'
import json
import math
import sys
from pathlib import Path

values = sorted(
    (float(v) for v in json.loads(Path(sys.argv[1]).read_text())["lambda_values"]),
    reverse=True,
)
selected = float(sys.argv[2])
matched = False
for value in values:
    if value + 1e-12 >= selected:
        print(f"{value:.15g}")
    if math.isclose(value, selected, rel_tol=0.0, abs_tol=1e-12):
        matched = True
        break
if not matched:
    raise SystemExit("selected lambda is not present in lambda grid")
PY
)

final_dir="$(printf '%s/%s/lambda_%.15g' "$output_base_dir" "$data_name" "$selected_lambda")"
final_result="$final_dir/result.json"
if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$final_result" ]; then
	echo "Skip existing result: $final_result"
	exit 0
fi

previous_result=""
for lambda_value in "${lambda_values[@]}"; do
	if "$uv_bin" run python - "$lambda_value" "$selected_lambda" <<'PY'
import math
import sys
raise SystemExit(0 if math.isclose(float(sys.argv[1]), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-12) else 1)
PY
	then
		output_dir="$final_dir"
		output_json="$final_result"
		is_final=1
	else
		output_dir="$(printf '%s/%s/lambda_%.15g' "$warm_base_dir" "$data_name" "$lambda_value")"
		output_json="$output_dir/result.json"
		is_final=0
	fi

	mkdir -p "$output_dir"
	temp_config="$output_dir/config.toml"
	cp "$config_template" "$temp_config"
	sed -i.bak "s/^lambda_fuse = .*/lambda_fuse = $lambda_value/" "$temp_config"
	rm -f "$temp_config.bak"

	command=(
		"$uv_bin" run python main.py
		--config "$temp_config"
		--data "$selected_data"
		--output "$output_json"
	)
	if [ "$is_final" -eq 1 ]; then
		command+=(--eval-data "$selected_eval")
	fi
	if [ -n "$previous_result" ]; then
		command+=(--init-result "$previous_result")
	fi
	"${command[@]}"
	previous_result="$output_json"
done

cp "$selection_file" "$final_dir/selected_lambda.json"
echo "Saved warm-path independent-evaluation result to: $final_result"
