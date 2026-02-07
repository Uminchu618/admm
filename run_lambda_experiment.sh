#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
data_dir="$repo_root/data/extended_aft_step"
output_base_dir="$repo_root/outputs/lambda_experiments"
config_template="$repo_root/config.toml"
lambda_grid_file="$repo_root/lambda_grid.json"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"

# データファイルリストを取得
mapfile -t data_files < <(ls "$data_dir"/*.csv 2>/dev/null | sort)
if [ "${#data_files[@]}" -eq 0 ]; then
	echo "No CSV files found in $data_dir" >&2
	exit 1
fi

# lambda値リストを取得
if [ ! -f "$lambda_grid_file" ]; then
	echo "Lambda grid file not found: $lambda_grid_file" >&2
	exit 1
fi

# jq で lambda_values 配列を抽出
mapfile -t lambda_values < <(jq -r '.lambda_values[]' "$lambda_grid_file")
if [ "${#lambda_values[@]}" -eq 0 ]; then
	echo "No lambda values found in $lambda_grid_file" >&2
	exit 1
fi

# SGE_TASK_ID を使って実験パターンを決定
# パターン数 = データ数 × lambda数
n_data="${#data_files[@]}"
n_lambda="${#lambda_values[@]}"
total_patterns=$((n_data * n_lambda))

# SGE_TASK_ID が未設定なら引数から取得（ローカル実行用）
if [ -z "${SGE_TASK_ID:-}" ]; then
	if [ $# -ge 1 ]; then
		SGE_TASK_ID="$1"
	else
		SGE_TASK_ID=1
	fi
fi

# 範囲チェック
if [ "$SGE_TASK_ID" -lt 1 ] || [ "$SGE_TASK_ID" -gt "$total_patterns" ]; then
	echo "SGE_TASK_ID out of range: $SGE_TASK_ID (1..$total_patterns)" >&2
	exit 1
fi

# データとlambdaのインデックスを計算（1-based → 0-based）
task_idx=$((SGE_TASK_ID - 1))
data_idx=$((task_idx / n_lambda))
lambda_idx=$((task_idx % n_lambda))

selected_data="${data_files[$data_idx]}"
selected_lambda="${lambda_values[$lambda_idx]}"

echo "=== Task $SGE_TASK_ID / $total_patterns ==="
echo "Data: $selected_data"
echo "Lambda: $selected_lambda"

# 出力ディレクトリ構造: outputs/lambda_experiments/{data_name}/lambda_{value}/
data_name="$(basename "$selected_data" .csv)"
lambda_dir="lambda_${selected_lambda}"
output_dir="$output_base_dir/$data_name/$lambda_dir"
mkdir -p "$output_dir"

# 一時config作成（lambda_fuseを上書き）
temp_config="$output_dir/config.toml"
cp "$config_template" "$temp_config"

# tomlファイルのlambda_fuse行を置換（簡易的にsedを使用）
sed -i.bak "s/^lambda_fuse = .*/lambda_fuse = $selected_lambda/" "$temp_config"

output_json="$output_dir/result.json"

cd "$repo_root"
"$uv_bin" run main.py \
	--config "$temp_config" \
	--data "$selected_data" \
	--output "$output_json"

echo "Saved result to: $output_json"
