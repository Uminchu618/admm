#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N support2

.venv/bin/python data/real/support/prepare_support2_inference.py \
  --input data/real/support/support2.csv \
  --output data/real/support/support2_inference.csv \
  --config config.toml

.venv/bin/python main.py \
  --data data/real/support/support2_inference.csv \
  --config config.toml \
  --output outputs/support2_result.json
