#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"

export DATA_DIR="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
export EVAL_DATA_DIR="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
export CV_OUTPUT_DIR="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv/$run_name}"
export OUTPUT_BASE_DIR="${PILOT_REFIT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refit/$run_name}"
export CONFIG_TEMPLATE="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/diagnostic_config.toml}"

exec "$repo_root/run_simulation_cv_refit.sh" "$@"
