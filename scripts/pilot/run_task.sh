#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

export DATA_DIR="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
export EVAL_DATA_DIR="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
export OUTPUT_BASE_DIR="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot}"
export CONFIG_TEMPLATE="${PILOT_CONFIG_TEMPLATE:-$repo_root/config.toml}"
export LAMBDA_GRID_FILE="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"

exec "$repo_root/run_lambda_experiment.sh" "$@"
