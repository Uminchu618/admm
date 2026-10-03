#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=4G
#$ -pe def_slot 2
#$ -tc 300
#$ -o logs/pilot_refined_cv
#$ -e logs/pilot_refined_cv

set -euo pipefail

# Array range is supplied by scripts/pilot/submit_refined_cv.sh.
./scripts/pilot/run_refined_cv_task.sh
