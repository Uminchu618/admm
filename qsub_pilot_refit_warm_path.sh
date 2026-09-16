#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=4G
#$ -pe def_slot 2
#$ -tc 100
#$ -o logs/pilot_refit_warm_path
#$ -e logs/pilot_refit_warm_path

set -euo pipefail

./scripts/pilot/run_refit_warm_path_task.sh
