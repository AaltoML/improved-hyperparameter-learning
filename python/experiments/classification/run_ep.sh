#!/bin/sh
#SBATCH --time=5:00:00
#SBATCH --mem=1G
#SBATCH --array=0-26

mkdir -p results/isotropic

python classification.py --model ep --dataset_id $SLURM_ARRAY_TASK_ID --ARD 0 --cv_seed $1