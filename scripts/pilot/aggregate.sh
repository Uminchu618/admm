#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-uv}"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
expected_datasets="${PILOT_EXPECTED_DATASETS:-100}"
output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv/$run_name}"
summary_path="${PILOT_SUMMARY_PATH:-$output_dir/cv_selections.csv}"
n_folds="${PILOT_N_FOLDS:-5}"
n_lambda="$(jq '.lambda_values | length' "$lambda_grid")"
expected_tasks=$((expected_datasets * n_lambda * n_folds))

completed="$(find "$output_dir" -name result.json -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Completed result.json: $completed / $expected_tasks"
if [ "$completed" -ne "$expected_tasks" ]; then
	echo "Pilot is incomplete; aggregation will include only completed tasks." >&2
fi

cd "$repo_root"
"$uv_bin" run scripts/simulation_cv/aggregate_results.py \
	--base-dir "$output_dir" \
	--n-folds "$n_folds" \
	--output "$summary_path"
