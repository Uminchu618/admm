#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-uv}"
run_name="${PILOT_RUN_NAME:-fixed_rho_newton5}"
refit_dir="${PILOT_REFIT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refit/$run_name}"
summary_path="${PILOT_REFIT_SUMMARY_PATH:-$refit_dir/refit_summary.csv}"

cd "$repo_root"
"$uv_bin" run scripts/aggregate_lambda_results.py \
	--base-dir "$refit_dir" \
	--output "$summary_path"

echo "Saved CV-selected independent-evaluation summary to: $summary_path"
