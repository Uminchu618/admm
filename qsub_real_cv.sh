#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=8G
#$ -pe def_slot 2
#$ -j y
#$ -N real_cv

#### Real-data CV: lambda 10 点 × 5-fold = 50 task
#### Framingham は qsub -v DATASET=framingham qsub_real_cv.sh のように投げる。
#$ -t 1-50:1
#$ -tc 50

#### 事前に split を 1 回だけ作成してください。
#### uv run scripts/real_cv/make_splits.py \
####   --dataset support2 \
####   --input data/real/support/support2.csv \
####   --output data/real/cv/splits/support2/support2_5fold_seed1234.csv \
####   --n-folds 5 \
####   --random-state 1234

./run_real_cv_experiment.sh
