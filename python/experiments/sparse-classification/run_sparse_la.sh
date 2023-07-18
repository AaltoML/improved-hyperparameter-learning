#!/bin/sh
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-3

mkdir -p results/isotropic

python sparse-la-ep.py --model la --dataset_id $SLURM_ARRAY_TASK_ID --ARD $1 --cv_seed $2 --exp_name $3 --num_inducing $4