#!/bin/sh
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --array=0-3

mkdir -p results/isotropic

python sparse-vi-epcvi.py --model tsvgp --dataset_id $SLURM_ARRAY_TASK_ID --ARD 0
