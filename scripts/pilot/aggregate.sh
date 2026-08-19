#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-uv}"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
expected_datasets="${PILOT_EXPECTED_DATASETS:-100}"
output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot/$run_name}"
summary_path="${PILOT_SUMMARY_PATH:-$repo_root/outputs/pilot/${run_name}_summary.csv}"
gate_path="${PILOT_GATE_PATH:-$repo_root/outputs/pilot/${run_name}_gate.json}"
n_lambda="$(jq '.lambda_values | length' "$lambda_grid")"
expected_tasks=$((expected_datasets * n_lambda))

completed="$(find "$output_dir" -name result.json -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Completed result.json: $completed / $expected_tasks"
if [ "$completed" -ne "$expected_tasks" ]; then
	echo "Pilot is incomplete; aggregation will include only completed tasks." >&2
fi

cd "$repo_root"
"$uv_bin" run scripts/aggregate_lambda_results.py \
	--base-dir "$output_dir" \
	--output "$summary_path"

"$uv_bin" run scripts/pilot/check_diagnostic.py \
	--summary "$summary_path" \
	--expected-rows "$expected_tasks" \
	--output-json "$gate_path"
