#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

export DATA_DIR="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
export EVAL_DATA_DIR="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"
export OUTPUT_BASE_DIR="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot/$run_name}"
export CONFIG_TEMPLATE="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/diagnostic_config.toml}"
export LAMBDA_GRID_FILE="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"

exec "$repo_root/run_lambda_experiment.sh" "$@"
