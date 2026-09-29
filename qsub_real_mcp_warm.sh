#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N real_mcp_warm
#$ -tc 5

./scripts/real_cv/run_mcp_warm_path.sh
