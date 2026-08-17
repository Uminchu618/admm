#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
run_name="${PILOT_DIAGNOSTIC_RUN:-adaptive_rho_newton5}"
output_dir="${PILOT_DIAGNOSTIC_OUTPUT_DIR:-$repo_root/outputs/pilot_diagnostic/$run_name}"
summary_path="${PILOT_DIAGNOSTIC_SUMMARY:-$repo_root/outputs/pilot_diagnostic/${run_name}_summary.csv}"
uv_bin="${UV_BIN:-uv}"

completed="$(find "$output_dir" -name result.json -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Completed diagnostic result.json: $completed / 54"
if [ "$completed" -ne 54 ]; then
	echo "Diagnostic run is incomplete; only completed tasks will be aggregated." >&2
fi

cd "$repo_root"
"$uv_bin" run scripts/aggregate_lambda_results.py \
	--base-dir "$output_dir" \
	--output "$summary_path"

"$uv_bin" run scripts/pilot/check_diagnostic.py \
	--summary "$summary_path" \
	--expected-rows 54 \
	--output-json "$repo_root/outputs/pilot_diagnostic/${run_name}_gate.json"
