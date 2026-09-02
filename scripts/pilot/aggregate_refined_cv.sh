#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"
coarse_cv_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv/$run_name}"
refined_dir="${PILOT_REFINED_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refined/$run_name}"
additions_dir="${PILOT_REFINED_ADDITIONS_DIR:-$repo_root/outputs/pilot_cv_refined_additions/$run_name}"
grid_path="${PILOT_REFINED_GRID:-$refined_dir/refined_grid.csv}"
n_folds="${PILOT_N_FOLDS:-5}"
uv_bin="${UV_BIN:-uv}"

cd "$repo_root"
"$uv_bin" run scripts/pilot/aggregate_refined_cv.py \
	--coarse-base-dir "$coarse_cv_dir" \
	--additions-base-dir "$additions_dir" \
	--grid "$grid_path" \
	--output-dir "$refined_dir" \
	--n-folds "$n_folds"

echo "Saved refined CV selections to: $refined_dir/cv_selections.csv"
