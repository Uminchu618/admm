#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-uv}"
train_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
eval_dir="${PILOT_EVAL_DIR:-$repo_root/data/pilot/eval}"
seed_start="${PILOT_SEED_START:-42}"
seed_end="${PILOT_SEED_END:-61}"

scenarios=(oracle fine_grid off_grid small no_change)

cd "$repo_root"
for scenario in "${scenarios[@]}"; do
	"$uv_bin" run generation/generate_extended_aft_step_datasets.py \
		--config "generation/pilot/${scenario}.json" \
		--output-dir "$train_dir" \
		--eval-output-dir "$eval_dir" \
		--seed-start "$seed_start" \
		--seed-end "$seed_end" \
		--prefix "${scenario}_seed_"
done

echo "Generated paired pilot data under:"
echo "  train: $train_dir"
echo "  eval:  $eval_dir"
