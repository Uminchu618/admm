#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
lambda_grid="${PILOT_DIAGNOSTIC_LAMBDA_GRID:-$repo_root/generation/pilot/diagnostic_lambda_grid.json}"
config_template="${PILOT_DIAGNOSTIC_CONFIG:-$repo_root/generation/pilot/diagnostic_config.toml}"
run_name="${PILOT_DIAGNOSTIC_RUN:-adaptive_rho_normalized_stagnation_escape_newton5}"
output_dir="${PILOT_DIAGNOSTIC_OUTPUT_DIR:-$repo_root/outputs/pilot_diagnostic/$run_name}"
uv_bin="${UV_BIN:-$(command -v uv)}"

for scenario in oracle fine_grid; do
	for seed in 42 43 44; do
		for required in "$train_dir/${scenario}_seed_${seed}.csv" "$eval_dir/${scenario}_seed_${seed}.csv"; do
			if [ ! -f "$required" ]; then
				echo "Required file not found: $required" >&2
				exit 1
			fi
		done
	done
done

mkdir -p "$repo_root/logs/pilot_diagnostic"
cd "$repo_root"
qsub -v "UV_BIN=$uv_bin,PILOT_TRAIN_DIR=$train_dir,PILOT_EVAL_DIR=$eval_dir,PILOT_DIAGNOSTIC_OUTPUT_DIR=$output_dir,PILOT_DIAGNOSTIC_CONFIG=$config_template,PILOT_DIAGNOSTIC_LAMBDA_GRID=$lambda_grid,PILOT_DIAGNOSTIC_RUN=$run_name" qsub_pilot_diagnostic.sh
