#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
action="${1:-}"
run_root="${PILOT_PENALTY_OUTPUT_ROOT:-$repo_root/outputs/pilot_penalty_comparison}"
lambda_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
lasso_config="${PILOT_LASSO_CONFIG:-$repo_root/generation/pilot/diagnostic_config.toml}"
mcp_config="${PILOT_MCP_CONFIG:-$repo_root/generation/pilot/mcp_config.toml}"
methods="${PILOT_PENALTY_METHODS:-lasso mcp}"

run_for_method() {
	local method="$1"
	local config="$2"
	local method_root="$run_root/$method"
	case "$action" in
		submit-cv)
			PILOT_RUN_NAME="$method" \
			PILOT_CONFIG_TEMPLATE="$config" \
			PILOT_LAMBDA_GRID="$lambda_grid" \
			PILOT_OUTPUT_DIR="$method_root/cv" \
			"$repo_root/scripts/pilot/submit.sh"
			;;
		submit-cv-warm)
			PILOT_CONFIG_TEMPLATE="$config" \
			PILOT_LAMBDA_GRID="$lambda_grid" \
			PILOT_OUTPUT_DIR="$method_root/cv" \
			"$repo_root/scripts/pilot/submit_warm_path.sh"
			;;
		aggregate-cv)
			PILOT_RUN_NAME="$method" \
			PILOT_LAMBDA_GRID="$lambda_grid" \
			PILOT_OUTPUT_DIR="$method_root/cv" \
			PILOT_SUMMARY_PATH="$method_root/cv/cv_selections.csv" \
			"$repo_root/scripts/pilot/aggregate.sh"
			;;
		submit-refit)
			PILOT_RUN_NAME="$method" \
			PILOT_CONFIG_TEMPLATE="$config" \
			PILOT_OUTPUT_DIR="$method_root/cv" \
			PILOT_REFIT_OUTPUT_DIR="$method_root/refit" \
			"$repo_root/scripts/pilot/submit_refit.sh"
			;;
		submit-refit-warm)
			PILOT_CONFIG_TEMPLATE="$config" \
			PILOT_LAMBDA_GRID="$lambda_grid" \
			PILOT_OUTPUT_DIR="$method_root/cv" \
			PILOT_REFIT_OUTPUT_DIR="$method_root/refit" \
			PILOT_REFIT_WARM_DIR="$method_root/refit_warm_paths" \
			"$repo_root/scripts/pilot/submit_refit_warm_path.sh"
			;;
		aggregate-refit)
			PILOT_RUN_NAME="$method" \
			PILOT_REFIT_OUTPUT_DIR="$method_root/refit" \
			PILOT_REFIT_SUMMARY_PATH="$method_root/refit/refit_summary.csv" \
			"$repo_root/scripts/pilot/aggregate_refit.sh"
			;;
	esac
}

case "$action" in
	submit-cv|submit-cv-warm|aggregate-cv|submit-refit|submit-refit-warm|aggregate-refit)
		for method in $methods; do
			case "$method" in
				lasso) run_for_method lasso "$lasso_config" ;;
				mcp) run_for_method mcp "$mcp_config" ;;
				*) echo "Unknown penalty method: $method" >&2; exit 2 ;;
			esac
		done
		;;
	compare)
		cd "$repo_root"
		uv run python scripts/pilot/compare_fused_penalties.py \
			--lasso-selections "$run_root/lasso/cv/cv_selections.csv" \
			--lasso-refits "$run_root/lasso/refit/refit_summary.csv" \
			--mcp-selections "$run_root/mcp/cv/cv_selections.csv" \
			--mcp-refits "$run_root/mcp/refit/refit_summary.csv" \
			--output-dir "$run_root/comparison"
		;;
	*)
		echo "Usage: $0 {submit-cv|submit-cv-warm|aggregate-cv|submit-refit|submit-refit-warm|aggregate-refit|compare}" >&2
		exit 2
		;;
esac
