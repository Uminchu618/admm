#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N real_full

#### CV-selected full fit: DATASETS=support2,framingham の各1 task = 2 task
#$ -t 1-2:1
#$ -tc 2

#### 既定では Support2 と Framingham の両方を投げます。
#### 片方だけ実行したい場合:
####   qsub -v DATASETS=support2 qsub_real_full.sh   # この場合 #$ -t は 1-1 に変更
####   qsub -v DATASETS=framingham qsub_real_full.sh # この場合 #$ -t は 1-1 に変更
####
#### 過去の全lambda/BICモード（LAMBDA_SELECTION_MODE=grid）の集計と可視化:
####   uv run scripts/real_cv/aggregate_full_results.py \
####     --base-dir outputs/real_full \
####     --output outputs/real_full/full_summary.csv
####   uv run scripts/real_cv/visualize_full_results.py \
####     --summary outputs/real_full/full_summary.csv \
####     --output-dir outputs/real_full/plots

./run_real_full_experiment.sh
