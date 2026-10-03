#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
data_dir="${PILOT_TRAIN_DIR:-$repo_root/data/pilot/train}"
output_base_dir="${PILOT_REFINED_ADDITIONS_DIR:-$repo_root/outputs/pilot_cv_refined_additions/adaptive_rho_normalized_stagnation_escape_newton5}"
manifest="${PILOT_REFINED_MANIFEST:?PILOT_REFINED_MANIFEST is required}"
config_template="${PILOT_CONFIG_TEMPLATE:-$repo_root/generation/pilot/diagnostic_config.toml}"
split_seed="${PILOT_SPLIT_SEED:-1234}"
n_folds="${PILOT_N_FOLDS:-5}"
uv_bin="${UV_BIN:-uv}"
task_id="${SGE_TASK_ID:-${1:-1}}"

for required in "$manifest" "$config_template"; do
	if [ ! -f "$required" ]; then
		echo "Required file not found: $required" >&2
		exit 1
	fi
done

task_record="$("$uv_bin" run python - "$manifest" "$task_id" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
task_id = int(sys.argv[2])
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = {int(row["task_id"]): row for row in csv.DictReader(handle)}
if task_id not in rows:
    raise SystemExit(f"Task ID {task_id} not found in {manifest}")
row = rows[task_id]
print(row["data_name"], row["lambda_fuse"], row["fold"], sep="\t")
PY
)"
IFS=$'\t' read -r data_name lambda_fuse fold <<<"$task_record"

data_path="$data_dir/$data_name.csv"
if [ ! -f "$data_path" ]; then
	echo "Training data not found: $data_path" >&2
	exit 1
fi
if [ "$fold" -lt 0 ] || [ "$fold" -ge "$n_folds" ]; then
	echo "Fold out of range: $fold (0..$((n_folds - 1)))" >&2
	exit 1
fi

output_dir="$(printf '%s/%s/lambda_%.15g/fold_%02d' "$output_base_dir" "$data_name" "$lambda_fuse" "$fold")"
output_json="$output_dir/result.json"
if [ -f "$output_json" ]; then
	if "$uv_bin" run python - "$output_json" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = payload.get("summary", {})
history = payload.get("history", {})
converged = summary.get("converged", history.get("converged", False))
score = summary.get("c_td_test", summary.get("c_td_eval", summary.get("c_td")))
raise SystemExit(0 if bool(converged) and math.isfinite(float(score)) else 1)
PY
	then
		echo "Skip reusable result: $output_json"
		exit 0
	fi
	backup_json="$output_dir/result.before_retry_${JOB_ID:-local}_${task_id}.json"
	mv "$output_json" "$backup_json"
	echo "Preserved previous unusable result as: $backup_json"
fi

mkdir -p "$output_dir"
temp_config="$output_dir/config.toml"
cp "$config_template" "$temp_config"
sed -i.bak "s/^lambda_fuse = .*/lambda_fuse = $lambda_fuse/" "$temp_config"
rm -f "$temp_config.bak"

echo "Refined CV task $task_id: $data_name lambda=$lambda_fuse fold=$fold"
cd "$repo_root"
"$uv_bin" run scripts/simulation_cv/prepare_fold.py \
	--data "$data_path" \
	--output-dir "$output_dir" \
	--fold "$fold" \
	--n-folds "$n_folds" \
	--random-state "$split_seed"

"$uv_bin" run main.py \
	--config "$temp_config" \
	--data "$output_dir/data/train.csv" \
	--eval-data "$output_dir/data/test.csv" \
	--output "$output_json"

echo "Saved refined CV result to: $output_json"
