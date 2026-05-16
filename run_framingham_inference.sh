#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N framingham_bp

.venv/bin/python main.py \
  --data data/real/framingham/framingham_inference.csv \
  --config config.toml \
  --output outputs/framingham_result_bp.json