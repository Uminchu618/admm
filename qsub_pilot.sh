#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=4G
#$ -pe def_slot 2
#$ -t 1-1200:1
#$ -tc 200
#$ -o logs/pilot
#$ -e logs/pilot

set -euo pipefail

./scripts/pilot/run_task.sh
