#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
data_dir="$repo_root/data/extended_aft"
output_dir="$repo_root/outputs/extended_aft"
config_path="$repo_root/config.toml"
uv_bin="${UV_BIN:-uv}"

mapfile -t files < <(ls "$data_dir"/*.csv 2>/dev/null | sort)
if [ "${#files[@]}" -eq 0 ]; then
	echo "No CSV files found in $data_dir" >&2
	exit 1
fi

selected_file=""
if [ $# -ge 1 ]; then
	if [[ "$1" =~ ^[0-9]+$ ]]; then
		idx="$1"
		if [ "$idx" -lt 1 ] || [ "$idx" -gt "${#files[@]}" ]; then
			echo "Index out of range: $idx (1..${#files[@]})" >&2
			exit 1
		fi
		selected_file="${files[$((idx-1))]}"
	else
		selected_file="$1"
	fi
else
	idx="${SGE_TASK_ID:-1}"
	if [ "$idx" -lt 1 ] || [ "$idx" -gt "${#files[@]}" ]; then
		echo "Index out of range: $idx (1..${#files[@]})" >&2
		exit 1
	fi
	selected_file="${files[$((idx-1))]}"
fi

mkdir -p "$output_dir"
base_name="$(basename "$selected_file" .csv)"
output_path="$output_dir/${base_name}.json"

cd "$repo_root"
"$uv_bin" run main.py --config "$config_path" --data "$selected_file" --output "$output_path"
