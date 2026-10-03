#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=4G
#$ -pe def_slot 2
#$ -tc 100
#$ -o logs/pilot_refit
#$ -e logs/pilot_refit

set -euo pipefail

# Array range is supplied by scripts/pilot/submit_refit.sh from dataset count.
./scripts/pilot/run_refit_task.sh
