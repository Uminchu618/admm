#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
run_name="${PILOT_RUN_NAME:-adaptive_rho_normalized_stagnation_escape_newton5}"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
coarse_grid="${PILOT_LAMBDA_GRID:-$repo_root/generation/pilot/lambda_grid.json}"
coarse_cv_dir="${PILOT_OUTPUT_DIR:-$repo_root/outputs/pilot_cv/$run_name}"
refined_dir="${PILOT_REFINED_OUTPUT_DIR:-$repo_root/outputs/pilot_cv_refined/$run_name}"
additions_dir="${PILOT_REFINED_ADDITIONS_DIR:-$repo_root/outputs/pilot_cv_refined_additions/$run_name}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/diagnostic_config.toml}"
cv_selections="${PILOT_CV_SELECTIONS:-$coarse_cv_dir/cv_selections.csv}"
grid_output="${PILOT_REFINED_GRID:-$refined_dir/refined_grid.csv}"
manifest="${PILOT_REFINED_MANIFEST:-$refined_dir/refined_task_manifest.csv}"
manifest_summary="${PILOT_REFINED_MANIFEST_SUMMARY:-$refined_dir/refined_task_manifest.json}"
n_folds="${PILOT_N_FOLDS:-5}"
split_seed="${PILOT_SPLIT_SEED:-1234}"
upper_extension="${PILOT_REFINED_UPPER_EXTENSION:-0.75}"
data_names="${PILOT_REFINED_DATA_NAMES:-}"
uv_bin="${UV_BIN:-$(command -v uv)}"

for required in "$coarse_grid" "$cv_selections" "$config_template"; do
	if [ ! -f "$required" ]; then
		echo "Required file not found: $required" >&2
		exit 1
	fi
done

generator_args=(
	--cv-selections "$cv_selections"
	--coarse-grid "$coarse_grid"
	--existing-base-dir "$coarse_cv_dir"
	--output-base-dir "$additions_dir"
	--grid-output "$grid_output"
	--manifest-output "$manifest"
	--summary-output "$manifest_summary"
	--n-folds "$n_folds"
	--upper-extension "$upper_extension"
)
if [ -n "$data_names" ]; then
	generator_args+=(--data-names "$data_names")
fi

cd "$repo_root"
"$uv_bin" run scripts/pilot/generate_refined_cv_manifest.py "${generator_args[@]}"
n_tasks="$(( $(wc -l < "$manifest") - 1 ))"
if [ "$n_tasks" -eq 0 ]; then
	echo "All refined CV tasks already have result.json. Nothing to submit."
	exit 0
fi

"$uv_bin" run python - "$manifest" "$train_dir" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
train_dir = Path(sys.argv[2])
with manifest.open(newline="", encoding="utf-8") as handle:
    data_names = sorted({row["data_name"] for row in csv.DictReader(handle)})
missing = [name for name in data_names if not (train_dir / f"{name}.csv").is_file()]
if missing:
    raise SystemExit(f"Missing training data for refined CV: {missing}")
PY

mkdir -p "$repo_root/logs/pilot_refined_cv"
qsub -t "1-$n_tasks:1" -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_REFINED_ADDITIONS_DIR=$additions_dir,PILOT_REFINED_MANIFEST=$manifest,PILOT_CONFIG_TEMPLATE=$config_template,PILOT_RUN_NAME=$run_name,PILOT_N_FOLDS=$n_folds,PILOT_SPLIT_SEED=$split_seed" qsub_pilot_refined_cv.sh
