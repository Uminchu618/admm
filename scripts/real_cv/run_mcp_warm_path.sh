#!/bin/bash
set -euo pipefail

# Optional validation experiment: one fold per task, descending lambda path.
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"
dataset="${DATASET:-support2}"
config_template="${CONFIG_PATH:-$repo_root/config_real_mcp.toml}"
lambda_grid="${LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
n_folds="${N_FOLDS:-5}"
split_seed="${SPLIT_SEED:-1234}"
experiment_name="${EXPERIMENT_NAME:-mcp_${n_folds}fold_seed${split_seed}_warm}"
splits_file="${SPLITS_FILE:-$repo_root/data/real/cv/splits/$dataset/${dataset}_${n_folds}fold_seed${split_seed}.csv}"
output_base="${OUTPUT_BASE_DIR:-$repo_root/outputs/real_cv/$dataset}"
input_csv="${REAL_CV_INPUT:-}"

if [ -z "$input_csv" ]; then
    case "$dataset" in
        support2) input_csv="$repo_root/data/real/support/support2.csv" ;;
        framingham) input_csv="$repo_root/data/real/framingham/framingham.csv" ;;
        *) echo "Unsupported DATASET: $dataset" >&2; exit 1 ;;
    esac
fi

task_id="${SGE_TASK_ID:-${1:-1}}"
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$n_folds" ]; then
    echo "Fold task out of range: $task_id (1..$n_folds)" >&2
    exit 1
fi
fold_idx=$((task_id - 1))

if [ ! -f "$splits_file" ] || [ ! -f "$lambda_grid" ]; then
    echo "Missing split file or lambda grid: $splits_file / $lambda_grid" >&2
    exit 1
fi

lambda_values=()
while IFS= read -r lambda_value; do
    lambda_values+=("$lambda_value")
done < <("$uv_bin" run python - "$lambda_grid" <<'PY'
import json
import math
import sys
from pathlib import Path

values = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["lambda_values"]
values = [float(value) for value in values]
if not values or len(values) != len(set(values)) or any(not math.isfinite(value) or value < 0 for value in values):
    raise SystemExit("lambda grid must contain distinct, finite, nonnegative values")
for value in sorted(values, reverse=True):
    print(f"{value:.15g}")
PY
)
if [ "${#lambda_values[@]}" -eq 0 ]; then
    echo "No lambda values found in $lambda_grid" >&2
    exit 1
fi

cd "$repo_root"
previous_result=""
for lambda_value in "${lambda_values[@]}"; do
    output_dir="$(printf '%s/%s/lambda_%.15g/fold_%02d' "$output_base" "$experiment_name" "$lambda_value" "$fold_idx")"
    result_path="$output_dir/result.json"
    if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "$result_path" ]; then
        previous_result="$result_path"
        continue
    fi
    "$uv_bin" run scripts/real_cv/prepare_fold.py \
        --dataset "$dataset" --input "$input_csv" --splits "$splits_file" \
        --config "$config_template" --fold "$fold_idx" \
        --lambda-fuse "$lambda_value" --output-dir "$output_dir"
    command=("$uv_bin" run main.py --config "$output_dir/config.json" \
        --data "$output_dir/data/train.csv" --eval-data "$output_dir/data/test.csv" \
        --output "$result_path")
    if [ -n "$previous_result" ]; then
        command+=(--init-result "$previous_result")
    fi
    "${command[@]}"
    previous_result="$result_path"
done
echo "Completed MCP warm path: $dataset fold=$fold_idx"
