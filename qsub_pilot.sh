#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=4G
#$ -pe def_slot 2
#$ -tc 200
#$ -o logs/pilot
#$ -e logs/pilot

set -euo pipefail

# Array range is supplied by scripts/pilot/submit.sh from datasets x lambdas.
./scripts/pilot/run_task.sh
