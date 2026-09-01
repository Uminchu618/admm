#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=4G
#$ -pe def_slot 2
#$ -t 1-54:1
#$ -tc 54
#$ -o logs/pilot_diagnostic
#$ -e logs/pilot_diagnostic

set -euo pipefail

./scripts/pilot/run_diagnostic_task.sh
