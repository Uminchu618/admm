#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"

datasets_raw="${DATASETS:-support2,framingham}"
config_template="${CONFIG_PATH:-$repo_root/config.toml}"
lambda_grid_file="${LAMBDA_GRID:-$repo_root/lambda_grid.json}"
selection_mode="${LAMBDA_SELECTION_MODE:-cv}"
experiment_name="${EXPERIMENT_NAME:-cv_selected_full}"
output_base_dir="${OUTPUT_BASE_DIR:-$repo_root/outputs/real_full}"
cv_output_base_dir="${CV_OUTPUT_BASE_DIR:-$repo_root/outputs/real_cv}"
cv_n_folds="${N_FOLDS:-5}"
cv_split_seed="${SPLIT_SEED:-1234}"
cv_experiment_name="${CV_EXPERIMENT_NAME:-}"

IFS=',' read -r -a datasets <<< "$datasets_raw"
if [ "${#datasets[@]}" -eq 0 ]; then
	echo "No DATASETS specified" >&2
	exit 1
fi

lambda_values=()
if [ "$selection_mode" = "grid" ]; then
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
elif [ "$selection_mode" != "cv" ]; then
	echo "Unsupported LAMBDA_SELECTION_MODE: $selection_mode (expected cv or grid)" >&2
	exit 1
fi

if [ -z "${SGE_TASK_ID:-}" ]; then
	if [ $# -ge 1 ]; then
		SGE_TASK_ID="$1"
	else
		SGE_TASK_ID=1
	fi
fi

n_datasets="${#datasets[@]}"
if [ "$selection_mode" = "cv" ]; then
	total_tasks="$n_datasets"
else
	n_lambda="${#lambda_values[@]}"
	total_tasks=$((n_lambda * n_datasets))
fi

if [ "$SGE_TASK_ID" -lt 1 ] || [ "$SGE_TASK_ID" -gt "$total_tasks" ]; then
	echo "SGE_TASK_ID out of range: $SGE_TASK_ID (1..$total_tasks)" >&2
	exit 1
fi

task_idx=$((SGE_TASK_ID - 1))
if [ "$selection_mode" = "cv" ]; then
	dataset_idx="$task_idx"
else
	dataset_idx=$((task_idx / n_lambda))
	lambda_idx=$((task_idx % n_lambda))
fi
dataset="${datasets[$dataset_idx]}"

selection_file=""
if [ "$selection_mode" = "cv" ]; then
	if [ -n "$cv_experiment_name" ]; then
		resolved_cv_experiment="$cv_experiment_name"
	else
		resolved_cv_experiment="${dataset}_${cv_n_folds}fold_seed${cv_split_seed}"
	fi
	selection_file="${CV_SELECTION_FILE:-$cv_output_base_dir/$dataset/$resolved_cv_experiment/selected_lambda.json}"
	if [ ! -f "$selection_file" ]; then
		echo "CV selection file not found: $selection_file" >&2
		echo "Run scripts/real_cv/aggregate_results.py first." >&2
		exit 1
	fi
	selected_lambda="$("$uv_bin" run python - "$selection_file" "$cv_n_folds" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_n_folds = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("selection_method") != "five_fold_cv_mean_c_td":
    raise SystemExit(f"Unexpected selection_method in {path}")
if int(payload.get("n_folds", -1)) != expected_n_folds:
    raise SystemExit(f"Expected {expected_n_folds} folds in {path}")
value = float(payload["selected_lambda"])
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"Invalid selected_lambda in {path}: {value}")
print(value)
PY
)"
else
	selected_lambda="${lambda_values[$lambda_idx]}"
fi

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
echo "Lambda selection: $selection_mode"
echo "Lambda: $selected_lambda"
if [ -n "$selection_file" ]; then
	echo "Selection file: $selection_file"
fi
echo "Output: $output_dir"

cd "$repo_root"

selection_args=()
if [ -n "$selection_file" ]; then
	selection_args=(--selection-file "$selection_file")
fi

"$uv_bin" run scripts/real_cv/prepare_full_data.py \
	--dataset "$dataset" \
	--input "$input_csv" \
	--config "$config_template" \
	--lambda-fuse "$selected_lambda" \
	"${selection_args[@]}" \
	--output-dir "$output_dir"

"$uv_bin" run main.py \
	--config "$output_dir/config.json" \
	--data "$output_dir/data/all.csv" \
	--output "$output_dir/result.json"

echo "Saved result to: $output_dir/result.json"
