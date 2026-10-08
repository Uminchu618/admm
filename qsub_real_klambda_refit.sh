#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N framingham_klambda_full
set -euo pipefail
: "${SEARCH_DIR:?Run klambda_workflow.sh submit-refit first}"
export OMP_NUM_THREADS="${NSLOTS:-2}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-2}"
export MKL_NUM_THREADS="${NSLOTS:-2}"
bash scripts/real_cv/klambda_workflow.sh refit
