#!/bin/bash
# SGE（Sun Grid Engine）ジョブスクリプト設定
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N real_bootstrap_ci

# エラーハンドリング：エラー時は即座に終了、未定義変数・パイプライン失敗で停止
set -euo pipefail

# 環境変数の設定（デフォルト値を含む）
# ブートストラップ反復回数（デフォルト: 200）
N_BOOTSTRAP=${N_BOOTSTRAP:-200}
# 並列ジョブ数（デフォルト: 1）
N_JOBS=${N_JOBS:-1}
# ランダムシード（デフォルト: 20260505）
RANDOM_STATE=${RANDOM_STATE:-20260505}

# ================================================================================
# Support2 データセットの前処理
# ================================================================================
# Support2データセットを推論用に変換し、設定に従って前処理を実行
.venv/bin/python data/real/support/prepare_support2_inference.py \
  --input data/real/support/support2.csv \
  --output data/real/support/support2_inference.csv \
  --config config.toml

# ================================================================================
# Support2 の基本結果ファイル確認と引数設定
# ================================================================================
# 既存の本推定結果があれば、点推定値と設定を再利用して元データの再 fit を省略
SUPPORT_BASE_ARG=()
if [ -f outputs/support2_result.json ]; then
  SUPPORT_BASE_ARG=(--base-result outputs/support2_result.json)
fi

# ================================================================================
# Framingham の基本結果ファイル確認と引数設定
# ================================================================================
# 優先順位: framingham_result_bp.json > framingham_result.json
FRAMINGHAM_BASE_ARG=()
if [ -f outputs/framingham_result_bp.json ]; then
  FRAMINGHAM_BASE_ARG=(--base-result outputs/framingham_result_bp.json)
elif [ -f outputs/framingham_result.json ]; then
  FRAMINGHAM_BASE_ARG=(--base-result outputs/framingham_result.json)
fi

# ================================================================================
# Support2 データセットのブートストラップ信頼区間計算
# ================================================================================
# ブートストラップ反復により、パラメータの信頼区間を計算
.venv/bin/python scripts/bootstrap_parameter_ci.py \
  --data data/real/support/support2_inference.csv \
  --config config.toml \
  "${SUPPORT_BASE_ARG[@]}" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --n-jobs "$N_JOBS" \
  --random-state "$RANDOM_STATE" \
  --output-json outputs/support2_bootstrap_ci.json

# ================================================================================
# Framingham データセットのブートストラップ信頼区間計算
# ================================================================================
# ブートストラップ反復により、パラメータの信頼区間を計算
.venv/bin/python scripts/bootstrap_parameter_ci.py \
  --data data/real/framingham/framingham_inference.csv \
  --config config.toml \
  "${FRAMINGHAM_BASE_ARG[@]}" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --n-jobs "$N_JOBS" \
  --random-state "$RANDOM_STATE" \
  --output-json outputs/framingham_bootstrap_ci.json
