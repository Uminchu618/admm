#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N real_full

#### Full real-data fit: DATASETS=support2,framingham × lambda_grid.json 10点 = 20 task
#### lambda_grid.json の点数や DATASETS を変える場合は #$ -t も合わせて変更する。
#$ -t 1-20:1
#$ -tc 20

#### 既定では Support2 と Framingham の両方を投げます。
#### 片方だけ実行したい場合:
####   qsub -v DATASETS=support2 qsub_real_full.sh   # この場合 #$ -t は 1-10 に変更
####   qsub -v DATASETS=framingham qsub_real_full.sh # この場合 #$ -t は 1-10 に変更
####
#### 集計と可視化:
####   uv run scripts/real_cv/aggregate_full_results.py \
####     --base-dir outputs/real_full \
####     --output outputs/real_full/full_summary.csv
####   uv run scripts/real_cv/visualize_full_results.py \
####     --summary outputs/real_full/full_summary.csv \
####     --output-dir outputs/real_full/plots

./run_real_full_experiment.sh
