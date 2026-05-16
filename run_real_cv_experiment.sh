#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"

dataset="${DATASET:-support2}"
input_csv="${REAL_CV_INPUT:-}"
config_template="${CONFIG_PATH:-$repo_root/config.toml}"
lambda_grid_file="${LAMBDA_GRID:-$repo_root/lambda_grid.json}"

n_folds="${N_FOLDS:-5}"
split_seed="${SPLIT_SEED:-1234}"
experiment_name="${EXPERIMENT_NAME:-${dataset}_${n_folds}fold_seed${split_seed}}"
splits_file="${SPLITS_FILE:-$repo_root/data/real/cv/splits/$dataset/${experiment_name}.csv}"
output_base_dir="${OUTPUT_BASE_DIR:-$repo_root/outputs/real_cv/$dataset}"

if [ -z "$input_csv" ]; then
	case "$dataset" in
		support2)
			input_csv="$repo_root/data/real/support/support2.csv"
			;;
		framingham)
			input_csv="$repo_root/data/real/framingham/framingham.csv"
			;;
		*)
			echo "Unsupported DATASET: $dataset" >&2
			exit 1
			;;
	esac
fi

if [ ! -f "$lambda_grid_file" ]; then
	echo "Lambda grid file not found: $lambda_grid_file" >&2
	exit 1
fi

if [ ! -f "$splits_file" ]; then
	echo "Split file not found: $splits_file" >&2
	echo "Create it once before qsub:" >&2
	echo "  $uv_bin run scripts/real_cv/make_splits.py --dataset $dataset --input $input_csv --output $splits_file --n-folds $n_folds --random-state $split_seed" >&2
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
total_patterns=$((n_lambda * n_folds))

if [ "$SGE_TASK_ID" -lt 1 ] || [ "$SGE_TASK_ID" -gt "$total_patterns" ]; then
	echo "SGE_TASK_ID out of range: $SGE_TASK_ID (1..$total_patterns)" >&2
	exit 1
fi

task_idx=$((SGE_TASK_ID - 1))
lambda_idx=$((task_idx / n_folds))
fold_idx=$((task_idx % n_folds))
selected_lambda="${lambda_values[$lambda_idx]}"

lambda_dir="$(printf 'lambda_%.15g' "$selected_lambda")"
fold_dir="$(printf 'fold_%02d' "$fold_idx")"
output_dir="$output_base_dir/$experiment_name/$lambda_dir/$fold_dir"

echo "=== Real-data CV task $SGE_TASK_ID / $total_patterns ==="
echo "Dataset: $dataset"
echo "Experiment: $experiment_name"
echo "Lambda: $selected_lambda"
echo "Fold: $fold_idx / $n_folds"
echo "Output: $output_dir"

cd "$repo_root"

"$uv_bin" run scripts/real_cv/prepare_fold.py \
	--dataset "$dataset" \
	--input "$input_csv" \
	--splits "$splits_file" \
	--config "$config_template" \
	--fold "$fold_idx" \
	--lambda-fuse "$selected_lambda" \
	--output-dir "$output_dir"

"$uv_bin" run main.py \
	--config "$output_dir/config.json" \
	--data "$output_dir/data/train.csv" \
	--eval-data "$output_dir/data/test.csv" \
	--output "$output_dir/result.json"

echo "Saved result to: $output_dir/result.json"
