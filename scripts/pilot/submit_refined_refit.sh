#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"

export PILOT_OUTPUT_DIR="${PILOT_REFINED_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refined/$run_name}"
export PILOT_REFIT_OUTPUT_DIR="${PILOT_REFINED_REFIT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refined_refit/$run_name}"

exec "$repo_root/scripts/pilot/submit_refit.sh"
