#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N real_bootstrap_ci

set -euo pipefail

N_BOOTSTRAP=${N_BOOTSTRAP:-200}
N_JOBS=${N_JOBS:-1}
RANDOM_STATE=${RANDOM_STATE:-20260505}

.venv/bin/python data/real/support/prepare_support2_inference.py \
  --input data/real/support/support2.csv \
  --output data/real/support/support2_inference.csv \
  --config config.toml

SUPPORT_BASE_ARG=()
if [ -f outputs/support2_result.json ]; then
  SUPPORT_BASE_ARG=(--base-result outputs/support2_result.json)
fi

FRAMINGHAM_BASE_ARG=()
if [ -f outputs/framingham_result_bp.json ]; then
  FRAMINGHAM_BASE_ARG=(--base-result outputs/framingham_result_bp.json)
elif [ -f outputs/framingham_result.json ]; then
  FRAMINGHAM_BASE_ARG=(--base-result outputs/framingham_result.json)
fi

.venv/bin/python scripts/bootstrap_parameter_ci.py \
  --data data/real/support/support2_inference.csv \
  --config config.toml \
  "${SUPPORT_BASE_ARG[@]}" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --n-jobs "$N_JOBS" \
  --random-state "$RANDOM_STATE" \
  --output-json outputs/support2_bootstrap_ci.json

.venv/bin/python scripts/bootstrap_parameter_ci.py \
  --data data/real/framingham/framingham_inference.csv \
  --config config.toml \
  "${FRAMINGHAM_BASE_ARG[@]}" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --n-jobs "$N_JOBS" \
  --random-state "$RANDOM_STATE" \
  --output-json outputs/framingham_bootstrap_ci.json
