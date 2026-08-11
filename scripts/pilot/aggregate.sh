#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-uv}"
output_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot}"
summary_path="${PILOT_SUMMARY_PATH:-$repo_root/outputs/pilot_summary.csv}"

completed="$(find "$output_dir" -name result.json -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Completed result.json: $completed / 1200"
if [ "$completed" -ne 1200 ]; then
	echo "Pilot is incomplete; aggregation will include only completed tasks." >&2
fi

cd "$repo_root"
"$uv_bin" run scripts/aggregate_lambda_results.py \
	--base-dir "$output_dir" \
	--output "$summary_path"
