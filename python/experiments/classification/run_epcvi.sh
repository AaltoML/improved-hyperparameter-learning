#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-26

mkdir -p results/isotropic

python classification.py --model epcvi --dataset_id $SLURM_ARRAY_TASK_ID --ARD 0 --cv_seed $1