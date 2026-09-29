#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"
config_path="${CONFIG_PATH:-$repo_root/config_real_mcp.toml}"
lambda_grid="${LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
output_root="${REAL_MCP_OUTPUT_ROOT:-$repo_root/outputs/real_cv}"
full_output_root="${REAL_MCP_FULL_OUTPUT_ROOT:-$repo_root/outputs/real_full}"
n_folds="${N_FOLDS:-5}"
split_seed="${SPLIT_SEED:-1234}"

action="${1:-}"
dataset="${2:-}"
if [ -z "$action" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 {submit|submit-warm|aggregate|baselines|visualize|submit-refit|plot-refit} {framingham|support2}" >&2
    exit 2
fi
case "$dataset" in
    framingham|support2) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

experiment="mcp_${n_folds}fold_seed${split_seed}"
warm_experiment="${experiment}_warm"
base_dir="$output_root/$dataset/$experiment"
warm_dir="$output_root/$dataset/$warm_experiment"
split_file="$repo_root/data/real/cv/splits/$dataset/${dataset}_${n_folds}fold_seed${split_seed}.csv"
dataset_output_base="$output_root/$dataset"
full_experiment="${experiment}_selected_full"

cd "$repo_root"

validate_inputs() {
    "$uv_bin" run python - "$config_path" "$lambda_grid" "$split_file" "$dataset" "$n_folds" <<'PY'
import json
import math
import sys
import tomllib
from pathlib import Path

import pandas as pd

config_path, grid_path, split_path = (Path(value) for value in sys.argv[1:4])
dataset, n_folds = sys.argv[4], int(sys.argv[5])
config = tomllib.loads(config_path.read_text(encoding="utf-8"))
if config.get("fuse_penalty") != "mcp":
    raise SystemExit("MCP config must set fuse_penalty=mcp")
if float(config["mcp_gamma"]) * float(config["rho"]) <= 1:
    raise SystemExit("MCP requires mcp_gamma * rho > 1")
values = [float(v) for v in json.loads(grid_path.read_text(encoding="utf-8"))["lambda_values"]]
if not values or len(values) != len(set(values)) or any(not math.isfinite(v) or v < 0 for v in values):
    raise SystemExit("lambda grid must contain distinct, finite, nonnegative values")
splits = pd.read_csv(split_path)
if not {"id", "fold"}.issubset(splits.columns):
    raise SystemExit("split file must have id and fold columns")
if splits["id"].duplicated().any() or set(splits["fold"]) != set(range(n_folds)):
    raise SystemExit("split file must assign each id once to folds 0..n_folds-1")
meta_path = Path(str(split_path) + ".meta.json")
meta = json.loads(meta_path.read_text(encoding="utf-8"))
if meta.get("dataset") != dataset or int(meta.get("n_folds", -1)) != n_folds:
    raise SystemExit("split metadata does not match dataset/fold count")
print(len(values))
PY
}

case "$action" in
    submit|submit-warm)
        n_lambda="$(validate_inputs | tail -n 1)"
        if [ "$action" = "submit" ]; then
            total_tasks=$((n_lambda * n_folds))
            qsub -t "1-${total_tasks}:1" -v "UV_BIN=$uv_bin,DATASET=$dataset,CONFIG_PATH=$config_path,LAMBDA_GRID=$lambda_grid,N_FOLDS=$n_folds,SPLIT_SEED=$split_seed,SPLITS_FILE=$split_file,EXPERIMENT_NAME=$experiment,OUTPUT_BASE_DIR=$dataset_output_base" qsub_real_mcp_cv.sh
        else
            qsub -t "1-${n_folds}:1" -v "UV_BIN=$uv_bin,DATASET=$dataset,CONFIG_PATH=$config_path,LAMBDA_GRID=$lambda_grid,N_FOLDS=$n_folds,SPLIT_SEED=$split_seed,SPLITS_FILE=$split_file,EXPERIMENT_NAME=$warm_experiment,OUTPUT_BASE_DIR=$dataset_output_base,SKIP_EXISTING=${SKIP_EXISTING:-0}" qsub_real_mcp_warm.sh
        fi
        ;;
    aggregate)
        "$uv_bin" run scripts/real_cv/validate_mcp_results.py \
            --base-dir "$base_dir" --lambda-grid "$lambda_grid" --n-folds "$n_folds"
        "$uv_bin" run scripts/real_cv/aggregate_results.py --base-dir "$base_dir" --n-folds "$n_folds"
        ;;
    baselines)
        "$uv_bin" run scripts/real_cv/compute_cox_baseline.py --base-dir "$base_dir"
        ;;
    visualize)
        [ -f "$base_dir/selected_lambda.json" ] || { echo "Run aggregate first: $base_dir" >&2; exit 1; }
        [ -f "$base_dir/cox_summary.csv" ] || { echo "Run baselines first: $base_dir" >&2; exit 1; }
        "$uv_bin" run scripts/real_cv/visualize_results.py \
            --base-dir "$base_dir" --summary-by-lambda "$base_dir/summary_by_lambda.csv" \
            --cox-summary "$base_dir/cox_summary.csv" --no-write-csv
        "$uv_bin" run scripts/real_cv/visualize_mcp_beta.py \
            --dataset "$dataset" --base-dir "$base_dir"
        ;;
    submit-refit)
        [ -f "$base_dir/selected_lambda.json" ] || { echo "Run aggregate first: $base_dir" >&2; exit 1; }
        qsub -t 1-1:1 -v "UV_BIN=$uv_bin,DATASETS=$dataset,CONFIG_PATH=$config_path,N_FOLDS=$n_folds,SPLIT_SEED=$split_seed,CV_EXPERIMENT_NAME=$experiment,CV_OUTPUT_BASE_DIR=$output_root,OUTPUT_BASE_DIR=$full_output_root,EXPERIMENT_NAME=$full_experiment" qsub_real_mcp_full.sh
        ;;
    plot-refit)
        [ -f "$base_dir/selected_lambda.json" ] || { echo "Run aggregate first: $base_dir" >&2; exit 1; }
        "$uv_bin" run scripts/real_cv/visualize_mcp_beta.py \
            --dataset "$dataset" --base-dir "$base_dir" \
            --full-dir "$full_output_root/$dataset/$full_experiment" --full-only
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
