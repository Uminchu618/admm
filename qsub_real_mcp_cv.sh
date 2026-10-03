#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N real_mcp_cv
#$ -tc 45

# The workflow supplies -t dynamically from the lambda-grid and fold counts.
./run_real_cv_experiment.sh
