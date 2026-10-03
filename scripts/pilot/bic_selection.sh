#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
action="${1:-}"
run_root="${PILOT_BIC_OUTPUT_ROOT:-$repo_root/outputs/pilot_bic_selection}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
lasso_config="${PILOT_LASSO_CONFIG:-$repo_root/generation/pilot/diagnostic_config.toml}"
mcp_config="${PILOT_MCP_CONFIG:-$repo_root/generation/pilot/mcp_config.toml}"
methods="${PILOT_PENALTY_METHODS:-mcp}"
expected_datasets="${PILOT_EXPECTED_DATASETS:-100}"
uv_bin="${UV_BIN:-uv}"

run_for_method() {
	local method="$1"
	local config="$2"
	local method_dir="$run_root/$method"
	local summary="$method_dir/summary.csv"
	local selected="$method_dir/bic_selected_records.csv"
	local analysis="$method_dir/analysis"
	case "$action" in
		submit)
			PILOT_BIC_OUTPUT_DIR="$method_dir" \
			PILOT_CONFIG_TEMPLATE="$config" \
			PILOT_LAMBDA_GRID="$lambda_grid" \
			PILOT_EXPECTED_DATASETS="$expected_datasets" \
			"$repo_root/scripts/pilot/submit_bic_warm_path.sh"
			;;
		aggregate)
			cd "$repo_root"
			"$uv_bin" run python scripts/aggregate_lambda_results.py \
				--base-dir "$method_dir" \
				--output "$summary" \
				--sort-by bic
			"$uv_bin" run python scripts/pilot/aggregate_bic_selection.py \
				--summary "$summary" \
				--base-dir "$method_dir" \
				--lambda-grid "$lambda_grid" \
				--output "$selected" \
				--audit-output "$method_dir/bic_selection_audit.csv" \
				--expected-datasets "$expected_datasets"
			;;
		visualize)
			cd "$repo_root"
			"$uv_bin" run python scripts/pilot/visualize_results.py \
				--summary "$summary" \
				--output-dir "$analysis"
			"$uv_bin" run python scripts/pilot/visualize_bic_selection.py \
				--selected "$selected" \
				--output-dir "$analysis"
			;;
	esac
}

case "$action" in
	submit|aggregate|visualize)
		for method in $methods; do
			case "$method" in
				lasso) run_for_method lasso "$lasso_config" ;;
				mcp) run_for_method mcp "$mcp_config" ;;
				*) echo "Unknown penalty method: $method" >&2; exit 2 ;;
			esac
		done
		;;
	*)
		echo "Usage: $0 {submit|aggregate|visualize}" >&2
		exit 2
		;;
esac
